# Monitor-reference inputs

The simulator does not call a language model while simulating agent-network
interactions. It reads the eight CSV files in `../prior/` instead. This
directory intentionally contains this explanation only; it does not ask users
to download another runtime input.

- `distribution_*.csv` contains `Score_Interval`, `Count_Benign`,
  `Count_Malicious`, and derived percentage/probability columns. The simulator
  uses the two count columns to construct the class-conditional score
  distributions.
- `deltaV_lookup_table_*.csv` contains `Prior_c_B`, `Prior_c_M`, and
  `Delta_V`. It is the matching Delta-V validation grid; the simulator checks
  it during initialization for the corresponding reference version.

These are derived aggregate inputs, not the original FormAI or WildJailbreak
records. The original datasets, prompts, monitor outputs, and access material
are intentionally not redistributed here.
