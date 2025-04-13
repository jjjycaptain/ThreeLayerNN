# 准备数据集
从 https://www.cs.toronto.edu/~kriz/cifar.html 下载cifar-10数据集（CIFAR-10 python version）,解压后将文件夹里的内容（不包含文件夹，只有数据）放到`./dataset/`目录下

# 下载最佳模型
从 https://drive.google.com/file/d/1PQrwE2CrUwzc5O9tEoj0wf5FF9mnhTMj/view?usp=sharing 下载最优模型，解压后将整个文件夹放在`./result/`目录下

# 测试最佳模型
运行以下脚本
```sh
python ./src/run_exp.py --test --load_model --model_dir ./result/ep100_bc64_lr0.001_hda256_hdb256_reg0.001_ld0.95_relu/
```
如果要测试你的模型，只需要将`model_dir`改为你的模型参数所在的目录就行，并将权重文件命名为`model.pkl`

# 训练
## 仅训练
运行以下脚本
```sh
python ./src/run_exp.py --train --num_epochs 100 --batch_size 64 --learning_rate 1e-3 --lr_decay 0.95 --hidden_dim1 256 --hidden_dim2 256 --reg 1e-3 --weight_scale 0.01 --activation relu
```
以上是我测试的最优模型超参数配置，要使用您自己的超参数配置，请修改对应参数，参数说明见`./src/run_exp.py`里的`parse_args()`函数
## 训练后直接测试
运行以下脚本
```sh
python ./src/run_exp.py --train --test --num_epochs 100 --batch_size 64 --learning_rate 1e-3 --lr_decay 0.95 --hidden_dim1 256 --hidden_dim2 256 --reg 1e-3 --weight_scale 0.01 --activation relu
```

# 超参数搜索
运行以下脚本，进行超参数网格搜索
```sh
python ./src/run_exp.py --param_search 2
```
要想修改超参数网格搜索的范围，请到`./src/param_search.py`的`grid_search`函数的开头修改。