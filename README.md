# Headline

> An awesome project.



# test1

# test 1.1

### test1.2



# test2

## test 2.1

### test 2.2

推流

```
git add .
git commit -m "更新说明" #提交到缓存

git remote add origin https://github.com/xty136694120/docsfile.git

git push -u origin master
```



## 退出ssh连接依旧保持程序运行

### 创建 screen 窗口

```shell
screen -S  name
# name可以设置为ssh、ftp，用于标注该 screen 窗口用途
# 示例：
screen -S ftp
```

> 注意，执行 screen -S name 之后系统会跳进一个新窗口，这个窗口就是我们创建的新进程（它来执行我们的命令）。
>
>  在这里面进行项目的启动即可。

### 退出保存

**Detached， 在窗口中键入Ctrl+a 键，再按下 d 键，就可以退出 SSH 登录，退出之后不会影响 screen 程序的执行（也就是说我们服务器上部署的项目就会一直运行了）。**



**回到原来状态**

```shell
screen -r
```



### 1查看screen 进程 – >
```shell
screen -ls
```



### 进入 screen 进程
如果只有一个 screen 进程，命令行输入 screen -r -d 即可
如果有多个screen, 我们可以通过它的 PID 进入screen,执行：

```shell
screen -r -d 1805
```




