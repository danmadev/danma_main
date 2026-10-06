//! Strict, bounded startup-only neuron file format. No network or shard side effects.
use danma_core::{Activation, Config, Neuron, MAX_DENDRITES_PER_NEURON};
use serde::{de, Deserialize, Deserializer};
use std::{collections::BTreeSet, fmt, fs::File, io::Read, marker::PhantomData, path::Path};

// Bounded startup limits sized to admit the single-node 784 -> 1024 -> 10 experiment.
// Runtime route-table and per-neuron dendrite bounds remain independent limits.
pub(crate) const MAX_FILE_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MAX_NEURONS: usize = 2_048;
pub(crate) const MAX_TOTAL_WEIGHTS: usize = 1_048_576;
const MAX_DURATION_MS: u64 = 600_000;
const MAX_LIVE_EVENTS: usize = 4_096;
const MAX_STALENESS_VERSIONS: u64 = 8;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    schema_version: u64,
    #[serde(deserialize_with = "object_only")]
    settings: Settings,
    #[serde(deserialize_with = "bounded_vec::<_, _, MAX_NEURONS>")]
    neurons: Vec<NeuronSpec>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    #[serde(deserialize_with = "finite_f32")]
    learning_rate: f32,
    activation_ttl_ms: u64,
    replay_retention_ms: u64,
    max_live_events: usize,
    max_staleness_versions: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NeuronSpec {
    id: u64,
    #[serde(deserialize_with = "finite_f32")]
    bias: f32,
    #[serde(deserialize_with = "activation")]
    activation: Activation,
    #[serde(deserialize_with = "bounded_vec::<_, _, MAX_DENDRITES_PER_NEURON>")]
    weights: Vec<Weight>,
}

fn activation<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Activation, D::Error> {
    match String::deserialize(deserializer)?.as_str() {
        "linear" => Ok(Activation::Linear),
        "relu" => Ok(Activation::Relu),
        _ => Err(de::Error::custom("activation must be linear or relu")),
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Weight {
    source: u64,
    #[serde(deserialize_with = "finite_f32")]
    weight: f32,
}

// Keep the same pre-narrowing check as the wire decoder, without changing its API.
fn finite_f32<'de, D: Deserializer<'de>>(deserializer: D) -> Result<f32, D::Error> {
    let value = f64::deserialize(deserializer)?;
    if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
        return Err(de::Error::custom("expected a finite f32"));
    }
    Ok(value as f32)
}

// Derived Serde structs also accept positional sequences; file v1 requires objects.
struct Object<T>(T);

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Object<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        object_only(deserializer).map(Self)
    }
}

fn object_only<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<T, D::Error> {
    struct ObjectVisitor<T>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>> de::Visitor<'de> for ObjectVisitor<T> {
        type Value = T;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            formatter.write_str("a JSON object")
        }

        fn visit_map<M: de::MapAccess<'de>>(self, map: M) -> Result<T, M::Error> {
            T::deserialize(de::value::MapAccessDeserializer::new(map))
        }
    }
    deserializer.deserialize_map(ObjectVisitor(PhantomData))
}

// Do not trust a deserializer's size hint or allocate an oversized sequence.
fn bounded_vec<'de, D, T, const LIMIT: usize>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    struct Bounded<T, const LIMIT: usize>(PhantomData<T>);
    impl<'de, T: Deserialize<'de>, const LIMIT: usize> de::Visitor<'de> for Bounded<T, LIMIT> {
        type Value = Vec<T>;

        fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
            write!(formatter, "an array with at most {LIMIT} elements")
        }

        fn visit_seq<S: de::SeqAccess<'de>>(self, mut seq: S) -> Result<Vec<T>, S::Error> {
            let mut values = Vec::new();
            while values.len() < LIMIT {
                match seq.next_element::<Object<T>>()? {
                    Some(value) => values.push(value.0),
                    None => return Ok(values),
                }
            }
            if seq.next_element::<de::IgnoredAny>()?.is_some() {
                return Err(de::Error::custom(format!("array exceeds {LIMIT} elements")));
            }
            Ok(values)
        }
    }
    deserializer.deserialize_seq(Bounded::<T, LIMIT>(PhantomData))
}

pub(crate) fn load(path: &Path) -> Result<Vec<Neuron>, String> {
    let file = File::open(path).map_err(|error| format!("cannot open neuron config: {error}"))?;
    read(file).map_err(|error| format!("neuron config {}: {error}", path.display()))
}

fn read(reader: impl Read) -> Result<Vec<Neuron>, String> {
    // Read limit+1, not metadata length: file growth and special files cannot bypass the cap.
    let mut bytes = Vec::new();
    reader
        .take((MAX_FILE_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|error| format!("cannot read neuron config: {error}"))?;
    if bytes.len() > MAX_FILE_BYTES {
        return Err(format!("file exceeds {MAX_FILE_BYTES} bytes"));
    }
    let config: Object<FileConfig> = serde_json::from_slice(&bytes)
        .map_err(|error| format!("invalid neuron config JSON: {error}"))?;
    config.0.into_neurons()
}

impl FileConfig {
    fn into_neurons(self) -> Result<Vec<Neuron>, String> {
        if self.schema_version != 1 {
            return Err("unsupported schema_version (expected 1)".into());
        }
        let settings = self.settings;
        if settings.learning_rate <= 0.0
            || settings.learning_rate > 1.0
            || !(1..=MAX_DURATION_MS).contains(&settings.activation_ttl_ms)
            || !(1..=MAX_DURATION_MS).contains(&settings.replay_retention_ms)
            || !(1..=MAX_LIVE_EVENTS).contains(&settings.max_live_events)
            || settings.max_staleness_versions > MAX_STALENESS_VERSIONS
        {
            return Err("settings outside file v1 bounds".into());
        }
        if self.neurons.is_empty() {
            return Err("at least one neuron is required".into());
        }
        let total_weights: usize = self.neurons.iter().map(|neuron| neuron.weights.len()).sum();
        if total_weights > MAX_TOTAL_WEIGHTS {
            return Err(format!("total weights exceeds {MAX_TOTAL_WEIGHTS}"));
        }
        // Validate the whole document before constructing any core neurons.
        let mut ids = BTreeSet::new();
        for neuron in &self.neurons {
            if neuron.id == 0 || !ids.insert(neuron.id) {
                return Err("zero or duplicate neuron ID".into());
            }
            let mut sources = BTreeSet::new();
            for weight in &neuron.weights {
                if weight.source == 0 || !sources.insert(weight.source) {
                    return Err(format!(
                        "zero or duplicate source ID in neuron {}",
                        neuron.id
                    ));
                }
            }
        }
        self.neurons
            .into_iter()
            .map(|neuron| {
                Neuron::new(
                    neuron.id,
                    neuron.bias,
                    neuron
                        .weights
                        .into_iter()
                        .map(|weight| (weight.source, weight.weight)),
                    Config {
                        activation: neuron.activation,
                        learning_rate: settings.learning_rate,
                        activation_ttl_ms: settings.activation_ttl_ms,
                        replay_retention_ms: settings.replay_retention_ms,
                        max_live_events: settings.max_live_events,
                        max_staleness_versions: settings.max_staleness_versions,
                    },
                )
                .map_err(|error| format!("invalid neuron {}: {error:?}", neuron.id))
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "neuron_config_tests.rs"]
mod tests;
