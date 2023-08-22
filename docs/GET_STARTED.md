# Getting Started
This page provides basic tutorials about the usage of TETer. For installation instructions, please see [INSTALL.md](INSTALL.md).

## Prepare Datasets

#### Download BDD100K
We present an example based on [BDD100K](https://bdd100k.com/) dataset. Please first download the images and annotations from the [official website](https://bdd-data.berkeley.edu/). 
For more details about the dataset, please refer to the [offial documentation](https://doc.bdd100k.com/download.html).

On the offical download page, the required data and annotations are

- `detection` set images: `100K Images`
- `detection` set annotations: `Detection 2020 Labels`
- `tracking` set images: `MOT 2020 Images`
- `tracking` set annotations: `MOT 2020 Labels`

#### Convert annotations

To organize the annotations for training and inference, we implement a [dataset API](../teter/datasets/parsers/coco_video_parser.py) that is similiar to COCO-style.

After downloaded the annotations, please transform the offical annotation files to CocoVID style as follows.

First, uncompress the downloaded annotation file and you will obtain a folder named `bdd100k`.


To convert the detection set, you can do as
```bash
mkdir data/bdd/annotations/det_20
python -m bdd100k.label.to_coco -m det -i bdd100k/labels/det_20/det_${SET_NAME}.json -o data/bdd/annotations/det_20/det_${SET_NAME}_cocofmt.json
```

To convert the tracking set, you can do as
```bash
mkdir data/bdd/annotations/box_track_20
python -m bdd100k.label.to_coco -m box_track -i bdd100k/labels/box_track_20/${SET_NAME} -o data/bdd/annotations/box_track_20/box_track_${SET_NAME}_cocofmt.json
```

The `${SET_NAME}` here can be one of ['train', 'val'].

Then, create a folder name `scalabel_gt` and move the `box_track_20` folder inside bdd100k to `data/bdd/annotations/scalabel_gt/`.

#### Symlink the data

It is recommended to symlink the dataset root to `$TETer/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.
Our folder structure follows

```
├── teter
├── tools
├── configs
├── data
│   ├── bdd
│   │   ├── images 
│   │   │   ├── 100k 
|   |   |   |   |── train
|   |   |   |   |── val
│   │   │   ├── track 
|   |   |   |   |── train
|   |   |   |   |── val
│   │   ├── annotations 
│   │   │   ├── box_track_20
│   │   │   ├── det_20
│   │   │   ├── scalabel_gt

```

#### Download TAO
a. Please follow [TAO download](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md) instructions.

b. Please also prepare the [LVIS dataset](https://www.lvisdataset.org/).

It is recommended to symlink the dataset root to `$TETer/data`.

If your folder structure is different, you may need to change the corresponding paths in config files.

Our folder structure follows

```
├── qdtrack
├── tools
├── configs
├── data
    ├── tao
        ├── frames
            ├── train
            ├── val
            ├── test
        ├── annotations
    ├── lvis
        ├── train2017
        ├── annotations    
```

### 2. Install the TETA API

For more details about the installation and usage of the TETA metric, please refer to [TETA](../teta/README.md).



### 3. Generate our annotation files

a. Generate TAO annotation files with 482 classes.
```shell
python tools/convert_datasets/tao2coco.py -t ./data/tao/annotations
```

b. Merge LVIS and COCO training sets.

Use the `merge_coco_with_lvis.py` script in [the offical TAO API](https://github.com/TAO-Dataset/tao/blob/master/scripts/detectors/merge_coco_with_lvis.py).

This operation follows the paper [TAO](https://taodataset.org/).

```shell
cd ${TAP_API}
python ./scripts/detectors/merge_coco_with_lvis.py --lvis ${LVIS_PATH}/annotations/lvis_v0.5_train.json --coco ${COCO_PATH}/annotations/instances_train2017.json --mapping data/coco_to_lvis_synset.json --output-json ${LVIS_PATH}/annotations/lvisv0.5+coco_train.json
```

You can also get the merged annotation file from [Google Drive](https://drive.google.com/file/d/1v_q0eWpKgVDMvmjQ8pBKPgHQQ8SLhLx0/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1XnwJ5FqsA_neV0MSXu42hg) (passcode: rkh2).

During the training and inference, we use an additional file which save all class names: [lvis_classes.txt](https://drive.google.com/file/d/1H9JHpUe6ZZdFCahKiaujjrsNO2X8ocN_/view?usp=sharing).
Please download it and put it in `${LVIS_PATH}/annotations/`.

## Run TETer
This codebase is inherited from [mmdetection](https://github.com/open-mmlab/mmdetection).
You can refer to the [offical instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/getting_started.md).
You can also refer to the short instructions below.
We provide config files in [configs](../configs).

### Train a model

#### Train with a single GPU

```shell
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

If you want to specify the working directory in the command, you can add an argument `--work-dir ${YOUR_WORK_DIR}`.

#### Train with multiple GPUs

```shell
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

Optional arguments are:

- `--no-validate` (**not suggested**): By default, the codebase will perform evaluation at every k (default value is 1, which can be modified like [this](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py#L174)) epochs during the training. To disable this behavior, use `--no-validate`.
- `--work-dir ${WORK_DIR}`: Override the working directory specified in the config file.
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--cfg-options 'Key=value'`: Overide some settings in the used config.

**Note**:

- `resume-from` loads both the model weights and optimizer status, and the epoch is also inherited from the specified checkpoint. It is usually used for resuming the training process that is interrupted accidentally.
- For more clear usage, the original `load-from` is deprecated and you can use `--cfg-options 'load_from="path/to/you/model"'` instead. It only loads the model weights and the training epoch starts from 0 which is usually used for finetuning.

#### Train BDD100k model

Load pretrained [QDTrack model](https://drive.google.com/file/d/1qkgnRt7XkL4cjkwB1qIRnOHiMaXXdXIu/view?usp=sharing) and train the CEM head on the BDD100K Detection dataset.
```angular2html
tools/dist_train.sh configs/bdd100k/cem_bdd.py 8 25000 --work-dir saved_models/teter_bdd/ --cfg-options load_from=saved_models/qdtrack-frcnn_r50_fpn_12e_bdd100k-13328aed.pth 
```

#### Train TAO model
First, train the detector and CEM on LVISv0.5 + COCO dataset.
```angular2html
tools/dist_train.sh configs/tao/cem_lvis.py 8 25000 --work-dir saved_models/cem_r101_lvis/
```
Then, train the instance appearance similarity head on the TAO dataset.
```angular2html
tools/dist_train.sh configs/tao/tracker_tao.py 4 25000 --work-dir saved_models/teter_r101_tao/ --cfg-options load_from=saved_models/cem_lvis/epoch_24.pth data.samples_per_gpu=4 
```


### Test a Model with COCO-format

Note that, in this repo, the evaluation metrics are computed with COCO-format.
But to report the results on BDD100K, evaluating with BDD100K-format is required.

- single GPU
- single node multiple GPU
- multiple node

Trained models for testing

- [BDD100K model](https://drive.google.com/file/d/1InuFZkOtIsYZLCe0HFK74YK-_a0X1q6q/view?usp=sharing)
- [TAO model](https://drive.google.com/file/d/17koyuCbnj42ioZRxZZ5DChmCoAaMwets/view?usp=sharing)

You can use the following commands to test a dataset.

```shell
# single-gpu testing
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--show] [--cfg-options]

# multi-gpu testing
./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [--out ${RESULT_FILE}] [--eval ${EVAL_METRICS}] [--cfg-options]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format. If not specified, the results will not be saved to a file.
- `EVAL_METRICS`: Items to be evaluated on the results. Allowed values depend on the dataset, e.g., `track`.
- `--cfg-options`: If specified, some setting in the used config will be overridden.

#### Test BDD100K model
```angular2html
tools/dist_test.sh configs/bdd100k/cem_bdd.py saved_models/teter_bdd_r50_1x_20220706_134248.pth 8 25000 --eval track --eval-options resfile_path=results/teter_bdd_results/
```
#### Test TAO model
```angular2html
tools/dist_test.sh configs/tao/tracker_tao.py saved_models/teter_tao_r101_2x_20220613_223321.pth 8 25000 --eval track --eval-options resfile_path=results/teter_tao_results/
```

### Conversion to the Scalabel/BDD100K format

We provide scripts to convert the output prediction into BDD100K format jsons and masks,
which can be submitted to BDD100K codalabs to get the final performance.


```shell
python tools/to_bdd100k.py ${CONFIG_FILE} [--res ${RESULT_FILE}] [--task ${EVAL_METRICS}] [--bdd-dir ${BDD_OUTPUT_DIR} --nproc ${PROCESS_NUM}] [--coco-file ${COCO_PRED_FILE}]
```

Optional arguments:
- `RESULT_FILE`: Filename of the output results in pickle format.
- `TASK_NAME`: Task names in one of [`det`, `ins_seg`, `box_track`, `seg_track`]
- `BDD_OUPPUT_DIR`: The dir path to save the converted bdd jsons and masks.
- `COCO_PRED_FILE`: Filename of the json in coco submission format.