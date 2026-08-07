# Member 3 contribution statement

Member 3 completed the MobileNetV2 transfer-learning component of the project:

- integrated torchvision MobileNetV2 with ImageNet pretrained weights and a 30-class classifier;
- implemented Frozen, Partial and Full parameter-control modes, while formally evaluating Frozen
  and Partial only;
- completed the Frozen-backbone experiment;
- initialized Partial fine-tuning from the locked Frozen checkpoint and trained blocks 15-18 plus
  the classifier with differential learning rates;
- made the shared Trainer accept an externally supplied model while retaining CustomCNN
  compatibility;
- added validation Top-5 and Macro-F1 reporting and configurable validation Macro-F1 checkpoint
  selection;
- preserved frozen BatchNorm layers in evaluation mode;
- produced resolved configs, histories, checkpoints, summaries and training curves;
- locked both checkpoints before test access and completed the single final test evaluation;
- produced test classification reports, 30 x 30 confusion matrices, plots and fixed error/source
  analysis materials.

Member 3 reused the locked Part 2 dataset, metadata, class mapping and splits prepared by the data
pipeline. Data cleaning, duplicate/conflict resolution and split generation are shared inputs and
must not be described as Member 3's independent contribution. CustomCNN/Baseline and other model
experiments likewise remain their respective members' work.
