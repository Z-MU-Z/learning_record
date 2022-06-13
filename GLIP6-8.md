# GLIP6.8

要使用的数据

GoldG, 0.8M
human-annotated gold grounding data curated by MDETR
[21], including Flickr30K, VG Caption [25], and GQA [18].

今天打算先把这个训练跑起来

ln -s  /data/coco/train2017     /home/muzhi/GLIP/DATASET/coco

ln -s /data/coco/annotations   /home/muzhi/GLIP/DATASET/coco

# LVIS

```jsx
python tools/test_grounding_net.py \
        --config-file ./configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
        --task_config ./configs/lvis/minival.yaml \
        --weight ./MODEL/glip_tiny_model_o365_goldg.pth \
        TEST.EVAL_TASK detection OUTPUT_DIR ./output3
        TEST.CHUNKED_EVALUATION 40  TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 4 TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 3000 MODEL.RETINANET.DETECTIONS_PER_IMG 300 MODEL.FCOS.DETECTIONS_PER_IMG 300 MODEL.ATSS.DETECTIONS_PER_IMG 300 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 300
```

'aerosol can. air conditioner. airplane. alarm clock. alcohol. alligator. almond. ambulance. amplifier. anklet. antenna. apple. applesauce. apricot. apron. aquarium. arctic . armband. armchair. armoire. armor. artichoke. trash can. ashtray. asparagus. atomizer. avocado. award. awning. ax. baboon. baby buggy. basketball backboard. backpack. handbag. suitcase. bagel. bagpipe. baguet. bait’ 

这里应该只是一部分吧

接着还是从这里讲起

```python
self.box_selector_test(box_regression, centerness, anchors,
                                       box_cls,
                                       token_logits,
                                       dot_product_logits,
                                       positive_map,
                                       )
```

这里比较关键的也是dot_product_logits 是 bs*(h*w)*256

也就是说之后的CLS结果并不来自于box_cls而是来自于如下这个函数

```python
scores = convert_grounding_to_od_logits_v2(
logits=dot_product_logits,
num_class=self.mdetr_style_aggregate_class_num,
positive_map=positive_map,
score_agg=self.score_agg,
disable_minus_one=False)
```

这里我们假设 score_agg 选用 “MEAN”

![Untitled](GLIP6%208%20002ddae0f813498c807638d548dfc5f8/Untitled.png)

Positive _map 这里其实就是记录了caption到分类的映射信息。因为有些词是由多个字构成的，因此我们选择取平均。

之后我们就可以得到score了

![Untitled](GLIP6%208%20002ddae0f813498c807638d548dfc5f8/Untitled%201.png)

COCO2017

```jsx
python tools/test_grounding_net.py --config-file ./configs/pretrain/glip_Swin_T_O365_GoldG.yaml --weight ./MODEL/glip_tiny_model_o365_goldg.pth \
TEST.IMS_PER_BATCH 4 \
MODEL.DYHEAD.SCORE_AGG "MEAN" \
TEST.EVAL_TASK detection \
MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS False \
OUTPUT_DIR ./output3 

```

```
python -m torch.distributed.launch --nproc_per_node=4 \
        tools/test_grounding_net.py \
        --config-file {config_file} \
        --task_config configs/lvis/minival.yaml \
        --weight {model_checkpoint} \
        TEST.EVAL_TASK detection OUTPUT_DIR {output_dir}
        TEST.CHUNKED_EVALUATION 40  TEST.IMS_PER_BATCH 4 SOLVER.IMS_PER_BATCH 4 TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM 3000 MODEL.RETINANET.DETECTIONS_PER_IMG 300 MODEL.FCOS.DETECTIONS_PER_IMG 300 MODEL.ATSS.DETECTIONS_PER_IMG 300 MODEL.ROI_HEADS.DETECTIONS_PER_IMG 300

```

# PreTrain

```
python -m torch.distributed.launch --nnodes 2 --nproc_per_node=16 tools/train_net.py \
    --config-file configs/pretrain/glip_Swin_T_O365_GoldG.yaml \
    --skip-test --use-tensorboard --override_output_dir {output_dir}

python  tools/train_net.py \
    --config-file configs/pretrain/glip_Swin_T_O365_mixedgrounding.yaml \
    --skip-test --use-tensorboard --override_output_dir ./train

```

nnodes：节点的数量，通常一个节点对应一个主机，方便记忆，直接表述为主机
node_rank：节点的序号，从0开始
nproc_per_node：一个节点中显卡的数量
-master_addr：master节点的ip地址，也就是0号主机的IP地址，该参数是为了让 其他节点 知道0号节点的位，来将自己训练的参数传送过去处理
-master_port：master节点的port号，在不同的节点上master_addr和master_port的设置是一样的，用来进行通信

'Are the flags to the right of the person that is pulled by the horse? Is there a lamp to the right of the animal on the left? price tag in antique shop for a clock. man in shiny metallic top hat.’

'person at the beach. the beach is sandy. Child wearing blue swimsuit. A small white surfboard. Does the bikini look blue? this is a surfboard. this is a barbie doll. Light blue body of water. the water is blue. the beach is tan. this is an outdoors scene.’

target 为ground truth 的标注框

[BoxList(num_boxes=8, image_width=1066, image_height=800, mode=xyxy), BoxList(num_boxes=16, image_width=837, image_height=560, mode=xyxy)]

## loss

```jsx
loss_box_cls, loss_box_reg, loss_centerness, loss_token, loss_contrastive_align, loss_dot_product_token, 
loss_shallow_contrastive = self.loss_evaluator(
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
```

![Untitled](GLIP6%208%20002ddae0f813498c807638d548dfc5f8/Untitled%202.png)

这里是指 ATSSLossComputation (maskrcnn_benchmark/modeling/rpn/loss.py)

BoxList(num_boxes=4080, image_width=535, image_height=480, mode=xyxy), BoxList(num_boxes=1020, image_width=535, image_height=480, mode=xyxy), BoxList(num_boxes=255, image_width=535, image_height=480, mode=xyxy), BoxList(num_boxes=72, image_width=535, image_height=480, mode=xyxy), BoxList(num_boxes=20, image_width=535, image_height=480, mode=xyxy)]

下面具体来看 ATSSLossComputation内部

```python
labels, reg_targets, token_labels, map_labels, gold_box_od_labels, od_label_of_tokens_labels, positive_indices = self.prepare_targets(targets, anchors,
                                                                             tokenized,
                                                                             positive_map,
                                                                             proj_tokens
                                                                             )
```

self.prepare_targets 从字面意思就是对target进行处理

例如 targets为8个box

这里的 positive_map 为 *8*256*

```python
torch.where(positive_map!=0)[0]
tensor([0, 1, 2, 3, 4, 5, 5, 6, 7])
torch.where(positive_map!=0)[1]
tensor([ 9,  3, 15, 26, 20, 31, 32, 38, 40])
相当于表明了每个object在caption中的位置。
例如55出现了两次，说明物体占了两个词
```

然后会在所有的anchor和target_box 之间计算ious和distance 选出一些候选的anchor,

并会记录下对应关系和value

token_labels_per_im  ([18134, 256]) 记录从anchor 到 token的映射

token_labels_per_im[anchors_to_gt_values == -INF] = unmatched_labels *#相当于是没有物体的就最后256放1*

总之最后返回的就是 cls_labels, reg_targets, token_labels这三项比较有用

 ([18134, 256]) 

```python
cls_loss = self.cls_loss_func(box_cls_flatten, labels_flatten.int()) / num_pos_avg_per_gpu
```

box_cls_flatten.shape
torch.Size([18134, 80])
labels_flatten.shape  sum为72类
torch.Size([18134])  

这个预测结果仍然是80类，但labels中仍有0和1

### dot_product_token_loss

```python
dot_product_token_loss = self.token_loss_func(dot_product_logits,
token_labels_stacked, text_masks=text_masks,
version="binary") / num_pos_avg_per_gpu
```

这里使用的是 token_sigmoid_binary_focal_loss  maskrcnn_benchmark/layers/sigmoid_focal_loss.py

```python
p = torch.sigmoid(pred_logits)
ce_loss = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction="none")
p_t = p * targets + (1 - p) * (1 - targets)
loss = ce_loss * ((1 - p_t) ** gamma)

if alpha >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
```

logits.shape
torch.Size([1, 18134, 256])
targets.shape
torch.Size([1, 18134, 256]) 两者的维度是一致的

接着 会根据 pos_inds  （72个候选）

```python
reg_loss = self.GIoULoss(box_regression_flatten, reg_targets_flatten, anchors_flatten,
weight=centerness_targets) / sum_centerness_targets_avg_per_gpu
centerness_loss = self.centerness_loss_func(centerness_flatten, centerness_targets) / num_pos_avg_per_gpu
```

这里就是在72个候选间去做了，至此就结束了训练过程。