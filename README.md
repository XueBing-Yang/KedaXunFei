# 科大讯飞实习经历
## 1.常用Linux 命令
```python
cp /cog15/wxliu11/liu/code/rag/KnowledgeBase/simple_rag/app_v2/deep_thought/get_data.py ./ # 在同一服务器上复制一个文件
scp -r xbyang13@172.31.169.49:/cog15/deep_thought/get_data.py ./ # 可以将远端服务器的文件下载到本地，需要输入密码
scp a.txt xbyang13@172.31.169.49:/cog15/deep_thought/ # 从本地到远程
nvidia-smi # 查看gpu卡
npu-smi info # 查看npu卡
git status # 查看本地文件和git仓库的区别
git restore src/hemt_design.py # 将本地文件恢复为git仓库保存的样子
git add . # 添加所有文件到缓存
git commit -m "提示词优化" # 为缓存的文件添加备注
git pull # 从仓库拉取代码并合并到当前分支到主分支，适用于单人开发
git pull --rebase # 多人开发
git push # 将缓存区的文件提交到仓库
unzip PINN4SOH-main.zip # 解压zip文件
nohup python main.py > logs/distill.log 2>&1 & #把 main.py 这个 Python 脚本在后台运行，并且把它的所有输出（包括报错）写进同一个日志文件，即使你退出终端也不会被杀掉。
ps aux | grep main.py # 找到PID
kill 1215044 # 杀死进程
docker ps # 查看容器列表
docker exec -it mineru-server-gpuo /bin/bash # 进入容器,exit退出
docker logs -f hemt_design # 查看容器日志
pwd # 查看当前挂在的区域
rm -f hemt.tar.gz # 删除文件
tensorboard --logdir=runs # 查看训练情况
pwd # 查看当前目录
rm file.txt # 删除文件
rm -r project # 递归删除文件夹下的文件
mv a.txt b.txt # 重命名或移动文件
head -n 10 a.txt # 显示文件的前10行
tail -n 10 a.txt # 显示文件的后10行
grep hemt a.txt # 在文件中搜索hemt字段
ps aux # 显示当前运行的进程
wget url # 下载文件
curl -O url #下载文件
df -h # 显示磁盘使用情况
du -sh # 查看当前目前的使用情况
```
