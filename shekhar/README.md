# Do Adversarial Patches Generalize? Attack Transferability Study Across Real-time Segmentation Models in Autonomous Vehicles

## Highlights

1) **New patch attack formulation**: We propose a variant of the original
EOT formulation for learning adversarial patches,
thus making it more realistic for a Real-time SS model
in autonomous driving scenarios.
2) **Adaptive adversarial training loss**: For learning the
patch itself, we proposed an adaptive loss function that
simplifies the one introduced before. In particular, we
reduce the number of hyper-parameters in the loss metric
to make the attack more robust.
3) **Comprehensive Transferability Study**: We analyze how
well adversarial patches transfer across different seg-
mentation models, identifying key architectural weak-
nesses. Additionally, by comparing CNN-based and
Transformer-based models, we provide insights into
their relative robustness against patch-based attacks. To
add to this, we also evaluate the per-class performance
degradation to determine which object categories (e.g.,
pedestrians, vehicles, buildings) are most affected by
adversarial patches. To the best of our knowledge, this
is the first research to compare the performance of
ViTs and CNN based SS models against patch based
adversarial attacks at this level of detail.
4) **Realistic Attack Scenarios**: Unlike some of the widely
used perturbation attack models such as FGSM, using
EOT here we propose an untargeted black-box adver-
sarial patch attack that is more realistic in a real-time
setting, since the attacker does not need access to the
SS model weights.

## Using the code
1. Place the pretrained models in the folder: pretrained_models
2. Edit the config.yaml in the configs folder to add root address of you dataset along with text file names at this address containing the directory address of each of the image and corresponding mask. Each row of these files will have the address of image and mask files separated by a space.
3. Finally go to Main.ipynb in Experiments folder to execute the code.

