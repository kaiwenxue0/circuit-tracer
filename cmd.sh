#1.  install transformer-lens locally for modifying the code
# 1) 只下载对应提交的源码包
mkdir -p remote && cd remote
curl -L -o tl.zip https://codeload.github.com/TransformerLensOrg/TransformerLens/zip/b5a16f849649a237cc02cc2c272ae4dc2085abe4

# 2) 解压并改名
unzip tl.zip
mv TransformerLens-b5a16f849649a237cc02cc2c272ae4dc2085abe4 transformer-lens
rm tl.zip
cd transformer-lens

# 3) 把这份源码初始化为仓库（以后能改、能推）
git init
git add -A
git commit -m "Base from upstream commit b5a16f8"

pip install -e .
# 可选：关联你的 fork（改成你的仓库地址）
# git remote add origin git@github.com:kaiwenxue0/TransformerLens.git
# git checkout -b my-dev
# git push -u origin my-dev

cd ../../        # 回到主项目根目录
git submodule add -b my-dev git@github.com:kaiwenxue0/TransformerLens.git remote/transformer-lens


# install circuit-tracer
pip install -e .