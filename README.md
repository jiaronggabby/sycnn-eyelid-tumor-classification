# SY-CNN eyelid tumour classification

Source code for the study of synthetic-image augmentation with conditional GAN and diffusion models for eyelid tumour classification. The repository contains code and the experiment configuration only. Clinical photographs, patient-level manifests, trained weights, generated images, and result files are not distributed.

The input dataset must be prepared and access-controlled separately. Set up the Python environment from `requirements.txt` together with a CUDA-compatible PyTorch and TorchVision installation, then follow the command-line help for `prepare_manual_eye_dataset.py` and `run_manual_eye_experiment.py`. Training is intended for a CUDA server; the runner checks the cohort manifests and patient-group split before execution.

The source includes the training pipeline, validation-based route selection, held-out evaluation, patient-group bootstrap analysis, and the training-set-only perceptual-neighbour analysis. No clinical data or model weights are required to browse this repository.
