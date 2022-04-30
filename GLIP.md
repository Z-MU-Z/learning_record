# GLIP



代码理解

GLIP/MODEL/glip_tiny_model_o365_goldg.pth



## 命令

```
hfai python tools/test_grounding_net.py --config-file /ceph-jd/pub/jupyter/zhumuzhi/notebooks/GLIP/configs/pretrain/glip_Swin_T_O365_GoldG.yaml --weight ./MODEL/glip_tiny_model_o365_goldg.pth \
        TEST.IMS_PER_BATCH 3 \
        MODEL.DYHEAD.SCORE_AGG "MEAN" \
        TEST.EVAL_TASK detection \
        MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
        OUTPUT_DIR ./output3 \
        -- --nodes 1 --priority 10
```

```
hfai logs -f tools/test_grounding_net.py 

hfai stop tools/test_grounding_net.py 
```





## 

## 改动

为了适配于集群目前不连外网的情况

from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE)

将之改为类似如下的形式

config = BertConfig.from_pretrained(self.bert_name,cache_dir=,local_files_only=True)

## **rpn.py** 

Module for RPN computation. Takes feature maps from the backbone and RPN proposals and losses. 

返回boxes以及loss

## **generalized_vl_rcnn.py**

定义了 class GeneralizedVLRCNN ，也就是整个模型的整体架构了

由以下三部分组成

1、backbone

2、  rpn 见 vldyhead.py

3、heads: takes the features + the proposals from the RPN and computes

​    detections / masks from it

例如在detection的任务下，caption为'person. bicycle. car. motorcycle. airplane. bus. train. truck. boat. traffic light. fire hydrant. stop sign. parking meter. bench. bird. cat. dog. horse. sheep. cow. elephant. bear. zebra. giraffe. backpack. umbrella. handbag. tie. suitcase. frisbee. skis. snowboard. sports ball. kite. baseball bat. baseball glove. skateboard. surfboard. tennis racket. bottle. wine glass. cup. fork. knife. spoon. bowl. banana. apple. sandwich. orange. broccoli. carrot. hot dog. pizza. donut. cake. chair. couch. potted plant. bed. dining table. toilet. tv. laptop. mouse. remote. keyboard. cell phone. microwave. oven. toaster. sink. refrigerator. book. clock. vase. scissors. teddy bear. hair drier. toothbrush'即类别组成的句子



### language_backbone

返回如下这些值

ret = {
            "aggregate": aggregate,
            "embedded": embedded,
            "masks": mask,
            "hidden": encoded_layers[-1]
        }

### visual_backbone

分两种，一种是单独的SWINT 只负责image feature

一种是

SWINT_VL

the backbone only update the "hidden" field, currently,对languge feature的hidden layer更新。

[2022-04-23 12:00:32.903059] swin out torch.Size([3, 96, 200, 304])
[2022-04-23 12:00:32.906505] swin out torch.Size([3, 192, 100, 152])
[2022-04-23 12:00:32.913978] swin out torch.Size([3, 384, 50, 76])
[2022-04-23 12:00:32.915966] swin out torch.Size([3, 768, 25, 38])

输出也就是把不同层的feature提取出来。

## vldyhead.py

也就是 rpn 部分

### VLFuse 

首先定义了 VLFuse 类

其内部主要就是一个attention层，用来对不同模态的类进行融合。

MHA-S 为单向只做T->I，MHA-B为双向

![image-20220423121933799](GLIP.assets/image-20220423121933799.png)

### VLDyHead 

应该是整个实现中最关键的地方了,



其内部的主体为一个dyhead_tower 由VLFuse ，language path 和 vision path三部分组成

并会根据需要使用的LOSS定义了一系列head

soft token head,contrastive alignment head, dot product soft token head.

如下是我目前使用的config中使用的Loss.

   USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS: False
      USE_CLASSIFICATION_LOSS: False
      USE_CONTRASTIVE_ALIGN_LOSS: False
      USE_DOT_PRODUCT_TOKEN_LOSS: True
      USE_FUSED_FEATURES_DOT_PRODUCT: True
      USE_LAYER_SCALE: True
      USE_SHALLOW_CONTRASTIVE_LOSS: False
      USE_SHALLOW_ZERO_PADS: False
      USE_TOKEN_LOSS: False

### VLDyHeadModule

对VLDyHead进行进一步封装,可观察它的输入和输出。

box_cls, box_regression, centerness, token_logits, \
        proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features = self.head(features,
                                                                        language_dict_features,
                                                                        embedding,
                                                                        swint_feature_c4
                                                                        )

print("box",box_cls[0].shape,box_regression[0].shape,centerness[0].shape)

box torch.Size([3, 80, 100, 152]) torch.Size([3, 4, 100, 152]) torch.Size([3, 1, 100, 152])

接下来就分为train和test

**train**会计算一系列Loss

loss_box_cls, loss_box_reg, loss_centerness, loss_token, loss_contrastive_align, loss_dot_product_token, loss_shallow_contrastive = self.loss_evaluator(
            box_cls, box_regression, centerness, targets, anchors,
            captions,
            positive_map,
            token_logits,
            proj_tokens,
            contrastive_logits,
            dot_product_logits,
            text_masks,
            shallow_img_emb_feats
        )



**test**

self.box_selector_test(box_regression, centerness, anchors,
                                       box_cls,
                                       token_logits,
                                       dot_product_logits,
                                       positive_map,
                                       )

这个centerness的含义是什么，另外box_regression目前还是有负值

接着还会产生anchors

 anchors = self.anchor_generator(images, features)

每张图会生成一些了的BoxLists

![image-20220426164723896](GLIP.assets/image-20220426164723896.png)





多模态的自监督

## Dataset

 Cannot find coco/annotations/instances_val2017.json in ['./', './DATASET', './OUTPUT', './data', './MODEL']

构建soft-link 使用public_dataset下的COCO

/public_dataset/1/COCO/annotations

ln -s /public_dataset/1/COCO/annotations  /ceph-jd/pub/jupyter/zhumuzhi/notebooks/GLIP/DATASET/coco

ln -s /public_dataset/1/COCO/val2017 /ceph-jd/pub/jupyter/zhumuzhi/notebooks/GLIP/DATASET/coco

## RESULT

最终的结果如下所示

![image-20220426194857540](GLIP.assets/image-20220426194857540.png)
