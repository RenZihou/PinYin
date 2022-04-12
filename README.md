# PinYIn

This is my homework #1 of AI course (30240042) in Tsinghua University.

***

## Usage

程序入口为`run.py`，该脚本接收4个命令行参数，使用方法如下：

```bash
python3 src/run.py \
  --task <stat|predict|val> \
  --model <bigram|trigram|quadgram> \
  --input /path/to/input/file \
  --output /path/to/output/file
```

其中，`task`参数默认为`predict`，`model`参数默认为`bigram`。
若要执行模型构建（词频统计），`task`选择`stat`，`input`传入语料库路径（支持文件夹或单文件），`output`传入模型文件输出路径；
若要运行输入法，`task`选择`predict`，`input`传入拼音文件路径，`output`传入输出文件路径，程序会自动加载`src/stat`目录下同名模型（因此请确保`stat/mapping.json`与`stat/<model>.json`文件存在）；
若要统计准确率，`task`选择`val`，`input`和`output`分别传入标准输出与程序输出文件路径（此时`model`参数无效）。
