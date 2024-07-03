_base_ = ['../_base_/models/classification/asc/asc.py',
          '../_base_/datasets/classification/asc.py', '../_base_/schedules/cls_epoch_schedule.py']

task_type = 'AscClsTask'
num_classes = {{_base_.num_classes}}

model = dict(num_classes=num_classes)

loss_func = dict(type='CrossEntropyLoss')
trainer = dict(max_epochs=350, check_val_every_n_epoch=1,devices=[1])
