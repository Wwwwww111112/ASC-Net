seed = 1234

task_type = 'BaseClsTask'

# 评估指标, 可为dict或list, MONAI提供的评估指标, 接口详情见MONAI官网
metrics = [dict(type='ConfusionMatrixMetric', include_background=False, reduction='mean', metric_name='accuracy'),
           dict(type='ROCAUCMetric')]

optims = dict(optimizer=dict(type='Adam', lr=0.0001),
              lr_scheduler=dict(
                  scheduler=dict(type='ReduceLROnPlateau', mode='min', factor=0.5, patience=10, min_lr=1e-4),
                  monitor='accuracy',
                  interval='epoch',
                  frequency=1
              )
              )

trainer = dict(type='Trainer',
               logger=[dict(type='TensorBoardLogger', version='tensorboard'), dict(type='CSVLogger', version='csv')],
               devices=[1],
               accelerator='gpu', strategy='auto', precision='32', max_epochs=100, check_val_every_n_epoch=1,
               callbacks=[dict(type='ModelCheckpoint', filename='{epoch}', every_n_epochs=1),
                          dict(type='ModelCheckpoint', monitor='accuracy', mode='max', save_top_k=1,
                               filename='{epoch}_{accuracy:.4f}'),
                          dict(type='LearningRateMonitor', logging_interval='step')], profiler='simple',
               num_sanity_val_steps=0)

ckpt_path = None
