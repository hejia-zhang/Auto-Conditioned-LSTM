### Reference
- [original paper](https://arxiv.org/abs/1707.05363)
- [original pytorch implementation](https://github.com/papagina/Auto_Conditioned_RNN_motion)

### Usage
- Clone the repo
- Generate training data
    
    ```bash
    # e.g.
    $ python data/generate_training_data --input ../train_data_bvh/martial --output_dir ../train_data_xyz/martial
    ```
- Training
    
    ```bash
    $ python EXP/aclstm_bvh.py
    ```