# nowcasting_DGMR

### How to use
- execute `python train.py`, then it will start to run.
- You can change the **base_channels** parameter to change the model size. The amount of model's weights is the same as paper when you set **base_channels** to 24.
- Now, I use random numbers to test if the model can be trained and run successfully. You can take a look at `DGMR_lightning.py` file.

### Some notes
- Implement [DGMR](https://arxiv.org/abs/2104.00954) model that is produced by Deep Mind to address radar nowcasting in meteorology.
- You can find more detail about the model architecture in [this supplementary](https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03854-z/MediaObjects/41586_2021_3854_MOESM1_ESM.pdf) provided by Deep Mind.
- Pytorch 1.10.2+cu102 is used in this implementation on dev-SuperRes server
- Other people's [implementation](https://github.com/openclimatefix/skillful_nowcasting)
