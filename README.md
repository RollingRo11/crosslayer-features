# crosslayer-features
Repo to train and visualize single-model crosscoders!

[Cross layer superposition](https://transformer-circuits.pub/2024/crosscoders/index.html) is a super interesting phenomenon! I wanted to build out a repo that allowed me to both train and visualize crosscoders in order to try to understand some of the weird things that happen to features as they're co-developed by multiple layers. 

**Currently, this repo only supports GPT2 (more support coming very soon!)**

A lot of the underlying crosscoder structure is from this [repo](https://github.com/neelnanda-io/Crosscoders/tree/main) from Neel Nanda!
(It's been adapted to use the NNSight library and more tailored to the research use here).

I create a visualization dashboard (inspired by a Goodfire tweet I saw a while ago).

<img width="1852" height="1180" alt="Screenshot 2025-10-14 at 4 20 38 PM" src="https://github.com/user-attachments/assets/126e4336-12e0-4500-8e04-4b9ab92b8b5f" />

# Usage
Pretty simple to setup and use:

#### Installation

Run `uv sync`

#### Usage
To train a crosscoder, first login with the huggingface CLI, then run `crosscoder/train.py --help`, and then set your desired hyperparameters with each argparse flag.

#### Visualization
To visualize crosscoder latents, run `generate_vis.py --help` and then set your desired hyperparameters with each argparse flag.

---

### Acknowledgements

- Thank you to Neel Nanda for being the (afaik) first person to upload their Crosscoder code. A lot of other implementations have been written based on his.
- Thank you to Claude Code for writing the visualization code that I didn't want to touch
- Thank you to Callum McDougall for the SAE_VIS repo + code! 
