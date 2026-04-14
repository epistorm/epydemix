# Making Computational Modeling Libraries Agent-Friendly: Lessons from Epydemix

## v1.1, 13/4/2026

## The general problem

Consider what an LLM agent faces when asked to "run an SEIR model of COVID-19 in Italy with school closures and compare three intervention timings." With a typical modeling library, the agent must read source code to discover which models exist, parse docstrings to determine which parameters are valid, guess at function signatures, retain intermediate objects in memory across tool calls, and somehow extract results from in-memory DataFrames that may contain millions of rows. Each of these steps is fragile. The agent has no structured way to discover capabilities, no way to validate a plan before executing it, and no compact way to read results. This is not a limitation of the agent. It is a common design limitation of modeling libraries.

## What we mean by "agent-friendly"

An agent-friendly library is one that a program — whether an LLM agent, a workflow orchestrator, or a CI pipeline — can drive without reading source code. Concretely, this requires five properties:

**Discoverability.** The agent can learn what the library offers (models, parameters, valid ranges, defaults) through structured queries rather than by reading source files.

**Declarative input.** The agent constructs a single configuration artifact (a YAML or JSON file) that fully specifies a modeling run. There is no sequence of imperative API calls to get right; the configuration is validated before execution.

**Statelessness.** Every operation is a self-contained function of its inputs: a configuration file plus, optionally, previously saved output files. No objects (e.g., Python objects) need to survive between tool calls, no session state accumulates in memory, and any step can be re-run independently. This is a key property that distinguishes an agent-friendly interface from a notebook-style API, in which the user builds a chain of mutable objects and calls methods on them in a specific order. An LLM agent's tool-use loop offers no reliable way to hold such objects across calls; a stateless approach sidesteps the problem entirely.

**Structured output.** Results are written to the filesystem in a self-describing format. The agent never needs to hold large objects in memory or parse unstructured text from the standard output of executables.

**Inspectability.** The agent can ask targeted questions about results ("when did infections peak?", "what was the attack rate?") and receive compact, structured answers without loading the full dataset.

None of these properties conflict with the interactive workflows (e.g., Jupyter Notebook) researchers already use. They are an additional layer on top of the existing API, and they also make the library better for human automation (shell scripts, CI/CD pipelines, reproducibility pipelines).

## The Epydemix library

Epydemix is a Python library for compartmental epidemic modeling. It supports SIR, SEIR, SIS, and fully custom compartmental models with age-structured populations, empirical contact matrices, time-varying parameters, and ABC-SMC calibration. It has a clean Python API: you instantiate an `EpiModel`, add compartments and transitions, set parameters, and call `run_simulations()`. Results come back as a dataclass holding pandas DataFrames and numpy arrays.

This API is well-designed for notebook and programmatic use, but from an agent's perspective, every run requires reconstructing a chain of imperative Python calls; there is no way to ask "what parameters does SEIR need?" without reading source code, and the results exist only as in-memory Python objects with no serialization. An agent cannot plan, validate, execute, and inspect a simulation without deep knowledge of the library's internals.

## Making Epydemix Agent-Friendly

We added five layers to Epydemix, each building on the previous one. The existing Python API was not modified — all changes are additive.

### 1\. Parameter registry

Every model parameter now has machine-readable metadata: name, description, data type, valid range, units, default value, and tags. A `ParameterRegistry` attached to each model can export this as JSON Schema. Predefined models (SIR, SEIR, SIS) automatically register their parameter specs. A catalog of defaults provides literature-sourced parameter sets for common diseases (COVID-19, influenza, measles), each with value ranges and citations.

The registry answers the agent's first question — "what knobs exist and what values are valid?" — without any source code reading.

### 2\. Structured output bundles

Simulation results are serialized to self-contained `.epx` directories containing Parquet files (compartment time-series, transition counts, parameter values) and a `manifest.json`. The manifest is small enough for an agent to read in full and contains the complete Parquet column schema, simulation metadata, and usage hints. This solves the data scale problem: a 1000-simulation run over 365 days with 16 age groups produces tens of millions of data points. The agent never touches the raw data directly; it reads the manifest to understand the schema, then uses inspection commands or writes targeted Parquet queries.

Bundles are the unit of reproducibility. Each includes a copy of the config that produced it, and figures are stored in a `figures/` subdirectory with manifest registration, so visualizations travel with the data.

### 3\. Inspection engine

A query layer sits between the agent and the Parquet files. Functions like `quantiles`, `summary`, `peak`, and `compare_bundles` read from the on-disk bundle, compute a compact answer, and return it as a dictionary (or JSON via the CLI). The agent asks "when did infections peak?" and gets a small JSON object, not a DataFrame with thousands or millions of rows.

A cross-bundle comparison engine computes standard epidemiological metrics (attack rate, peak timing, peak magnitude, total deaths, days over a threshold) across multiple scenario bundles in a single call, with optional baseline-delta computation.

### 4\. CLI entry point

A command-line interface, built with Click, provides the tool-use surface agents need. The design follows a strict convention: structured JSON to standard output, diagnostics to standard error, meaningful exit codes. The main commands are:

- `epydemix models` / `schema` / `defaults` / `populations` — discovery  
- `epydemix validate config.yaml` — pre-execution validation  
- `epydemix run config.yaml -o results.epx` — execution  
- `epydemix inspect results.epx <command>` — result queries  
- `epydemix compare *.epx` — cross-scenario comparison

The CLI is configuration-driven: the agent constructs a YAML file, validates it, and submits it. Configuration inheritance with deep-merge semantics lets scenario sweeps share a single base config, with each scenario overlay specifying only the parameters that change. Circular references are detected, chains are resolved at load time, and the resulting configuration is transparent to the rest of the pipeline.

#### Scheduled transitions

Vaccination campaigns, prophylaxis roll-outs, and staged treatment programs move individuals between compartments according to an external time-varying schedule: a certain number of doses per day per demographic group, independent of the current disease state. These "scheduled transitions" are challenging for declarative configuration: the dose schedule is typically a time series that must be aligned to the simulation's timeline and broadcast across demographic groups.  
The configuration layer supports this through `kind: scheduled` transitions. The agent specifies a `schedule` field — either a CSV file path (date-indexed, with one column per demographic group) or an inline list — and an optional `eligible` field that lists the compartments that receive doses. The schedule loader aligns the dose array with the simulation timeline and broadcasts single-column schedules across all demographic groups.  
The `eligible` mechanism warrants specific attention: in a mass vaccination campaign, doses are administered to individuals regardless of their disease state — a vaccinator cannot distinguish between susceptible and recovered individuals. When `eligible: ["S", "R"]` is specified, the denominator is the combined S+R population, so the effective rate on the source compartment (S) correctly reflects dose-wasting on individuals who are already immune. An agent is unlikely to get this right from first principles; making it a single configuration field eliminates the risk.

#### Declarative calibration

Epydemix already has a capable calibration engine, but it requires substantial imperative setup: the user needs to write a simulation wrapper function, construct scipy.stats distribution objects for priors, load and preprocess observed data, instantiate an `ABCSampler`, and extract results from nested dictionaries. An agent would have a hard time working with all of these internal conventions.  
We made calibration fully configuration-driven by extending the same YAML format used for simulation. A `calibration` section specifies priors (as declarative distribution specs translated to scipy.stats objects at load time), a pointer to observed data (either a CSV file path or inline values), the target model variable, a loss/distance function name, the ABC strategy, and strategy-specific settings. The agent writes a YAML file and calls `epydemix calibrate config.yaml`; the CLI handles the rest.  
The key design challenge is the simulation function. The ABC sampler expects a callable that takes a parameter dictionary and returns `{"data": array}`. In interactive Python, the user writes this wrapper by hand. For the CLI, we auto-generate it from the configuration: the wrapper builds the model once, then on each call updates the calibrated parameters, runs a single simulation, and extracts the target variable. This is invisible to the agent — it just specifies which variable to fit against.  
Prior distributions are specified declaratively, mapping a small vocabulary of distribution names (uniform, normal, truncnorm, beta, gamma, lognormal, exponential) to their natural parameters. This covers the distributions commonly used in epidemiological calibration without requiring the agent to know Scipy's parameterization conventions (which are somewhat unintuitive, e.g., Scipy's `uniform(loc, scale)` ).  
Calibration configurations support inheritance: a base configuration defines the model, population, and simulation settings; a calibration overlay adds only the `calibration` section. This means that the same base configuration can be used for both forward simulation and calibration, and multiple calibration configurations (different priors, different observed datasets) can share a single model definition.  
Calibration bundles extend the simulation bundle format with additional Parquet files for posterior distributions and loss/distance values per generation. The `inspect posterior` and `inspect fit` commands read from these files and return compact JSON summaries — posterior means, medians, credible intervals, and simulated-vs-observed trajectories.

#### Projections from posteriors

The existing `ABCSampler.run_projections()` method handles projections in Python, but it requires holding the sampler object in memory and calling methods imperatively. This is the kind of stateful approach that agents cannot drive.  
We made projections configuration-driven and stateless. A new `epydemix project` command takes a calibration bundle and an optional configuration overlay, reads the posterior and particle weights from the bundle's Parquet files, samples parameter sets weighted by importance, runs forward simulations with the (possibly overridden) configuration, and saves results as a standard simulation bundle. Because the output is a standard simulation bundle, it reuses the same `inspect quantiles`/`summary`/`peak` and `compare` commands as any other simulation run — no special inspection method is needed for projections.  
The key design choice was to entirely decouple projections from the `ABCSampler`. Rather than reconstructing the sampler object (which would require re-instantiating the simulation function, priors, and distance function), the projection pipeline operates directly on the saved artifacts: `posterior.parquet` for parameter values, `weights.parquet` for sampling probabilities, and the stored `config.yaml` for the model definition. This makes projections a purely functional operation on the bundle contents — reproducible, portable, and trivially parallelizable.  
Configuration inheritance is essential here: a projection overlay specifies `base: calibration.epx/config.yaml` and changes only what differs — typically the simulation end date, new interventions, or parameter overrides. The model definition, initial conditions, and all other settings are inherited from the calibration run. Multiple projection scenarios (baseline, early intervention, late intervention) can share the same calibration bundle with different overlays, and the resulting simulation bundles can be compared with `epydemix compare`.

### 5\. Agent contract document

An `AGENT.md` file at the repository root serves as the complete interface contract between Epydemix and any LLM agent driving it.

It opens with the discovery-to-analysis workflow — the canonical sequence of commands from "I have a disease description" to "I have a comparison table across scenarios" — so that the agent has a backbone connecting everything else, before diving into specifics. From there it documents the configuration format with every field annotated (type, allowed values, defaults, relationship to other fields), the full output bundle structure (what files exist, what columns they contain, what the manifest records), every inspection and compare subcommand with example inputs and the exact JSON shape of their outputs, and a translation guide for converting natural-language disease descriptions ("SEIR with waning immunity and a vaccination campaign") into the compartment/transition language. Each section is self-contained: an agent that needs to run a comparison does not have to read the calibration section first.

We chose to include worked examples rather than only schemas. A JSON Schema tells the agent what fields are allowed; a worked example tells it what a real configuration looks like, which fields are typically set together, and what the resulting output looks like. Agents generalize from examples far more reliably than they interpolate from abstract specifications, and the marginal context cost of a handful of concrete runs is small compared to the correctness gain.

This document is what an agent reads first and, ideally, last. If the agent needs to read source code to complete a task that falls within the documented interface, the contract has failed: either the interface is underspecified, the examples are unrepresentative, or the workflow has a gap. For the agent, reading Epydemix’s source code should be a last-resort action. We treat those failures as bugs in `AGENT.md` itself, and the document is versioned and updated alongside the code it describes.

Visualization recipes, with ready-to-run Python that loads the bundle and produces publication-quality plot, and workflow examples are provided in a separate `AGENT_EXAMPLES.md`, which is referenced from `AGENT.md`.

## Design decisions

**Three-tier output architecture.** The critical insight is that simulation output is too large for any agent to consume directly. The three-tier design (opaque Parquet bundle, small JSON manifest, compact CLI answers) lets the agent work at whatever level of detail it needs, from a one-line summary to a full custom pandas analysis. The manifest is the bridge: small enough to fit in a context window, detailed enough to write correct Parquet queries against.

**Config inheritance over templating.** Scenario sweeps require running the same model with systematic parameter variations. Rather than a templating system (Jinja, string interpolation), we implemented configuration inheritance with deep-merge semantics. A child configuration specifies `base: parent.yaml` and overrides only the fields that differ. Dictionaries are merged recursively; lists and scalars are replaced. This means an intervention overlay can fully specify its own intervention list without appending to or modifying the parent's. The result is that scenario configurations are small and readable.

**Metric registry for comparison.** Cross-scenario comparison is a universal need, but the specific metrics depend on the model structure (which compartment is "infected"? is there a death compartment?). Rather than requiring the user to specify compartment mappings, the comparison engine uses heuristic column detection — it finds S-like columns for attack rate, I-like columns for peak, and D-like columns for deaths — and gracefully falls back when heuristics fail. This makes `epydemix compare` work across SIR, SEIR, and custom models without configuration.

**Externalizing state into files, not objects.** The core design decision was to make every operation a stateless function of files. In the original library, a typical workflow builds up a chain of mutable Python objects (a model, a sampler, a results container) and calls methods on them in sequence. This works in a notebook, where the kernel holds objects in memory between cells, but it is hostile to agents, whose tool-use loops provide no reliable way to persist Python objects across calls. We externalize all intermediate state into self-contained “bundles”: a simulation produces a bundle, a calibration produces a bundle, and a projection reads a calibration bundle and produces a new simulation bundle. No step requires any object from a previous step to survive. This is why projections are decoupled from `ABCSampler` (they read the posterior and weights from Parquet files rather than calling methods on a live sampler) and why scheduled transitions embed their dose schedules as arrays in the model at config load time rather than referencing an external schedule object.

**Figures inside bundles.** Visualizations are stored in `<bundle>/figures/` and registered in the manifest. This keeps bundles self-contained (data \+ config \+ figures live together) and self-describing (the manifest records what each figure shows and which variables it covers).

**Scheduled transitions as configurations, not code.** Dose-driven transitions, such as vaccination campaigns, prophylaxis rollouts, and staged treatments, introduce a coupling between the model and an external time series that standard rate-based transitions lack. In the Python API, a user constructs a numpy array of daily doses, optionally specifies eligible compartments, and passes both to `add_transition()`. This is straightforward in a notebook but requires the agent to manage array construction, date alignment, and demographic group broadcasting — the sort of low-level data wrangling that is prone to bugs. The configuration layer handles all of this declaratively: the user specifies `kind: scheduled` with a `schedule` field (a CSV path or inline list) and an optional `eligible` field, and the builder loads, aligns, broadcasts, and wires the dose array into the model. The `eligible` mechanism is worth highlighting because it encodes a domain-specific subtlety — dose-wasting when a campaign vaccinates both susceptible and recovered individuals — as a single configuration field rather than leaving it to the agent's reasoning.

**Lightweight provenance through manifest lineage.** In an agent-driven workflow, the chain of operations — calibrate, project baseline, project with intervention, compare — can be long and branching. An agent's memory is bounded by its context window, and a bundle in the filesystem carries no trace of how it was produced beyond its stored configuration. We added a `provenance` key to every manifest that records the command that created the bundle, the configuration file path, and — for projections — the parent calibration bundle. This is deliberately lightweight: it captures lineage (what produced this bundle and from what inputs) without attempting a full audit trail of every parameter explored or rejected. The choice was informed by the observation that the most common provenance question is "where did this come from?", not "what else was tried." The configuration file and parent bundle pointer answer the first question; the agent's own conversation transcript answers the second.

**CLI over MCP.** A natural alternative to a command-line interface would be an Model Context Protocol (MCP) server: a structured tool-use interface where each operation is a named tool with a typed schema. We believe that command line interface (CLI) tools are preferable for several reasons: First, the hard problem is not the transport layer but the statelessness of the underlying operations. An MCP server wrapping a stateful library faces the same challenge we faced: either the server holds Python objects in memory between calls (introducing session management, crash recovery, and the same imperative ordering constraints agents struggle with), or it makes each call self-contained — in which case it is doing exactly what our CLI does. Second, CLI tools provide the most universal interface (any agent framework can invoke shell commands), and an MCP server would be just a thin wrapper over the same functions. Third, a CLI is human-inspectable. A researcher can type `epydemix inspect results.epx posterior` in a terminal and see exactly what the agent sees. An MCP server interposes a protocol layer that makes this harder. Fourth, the CLI approach requires no additional infrastructure: no server process, no authentication, etc.

**Backward compatibility.** Every addition we made was designed to be backward-compatible. The parameter registry, output bundles, CLI, and config system are optional layers. Existing code that calls `model.add_parameter("beta", 0.3)` and works with in-memory results continues to work unchanged. This is important in a research setting where existing notebooks, scripts, and published code on the current API.

