export https_proxy=http://agent.baidu.com:8891

git remote -v
git remote set-url origin ssh://xiafeifan@icode.baidu.com:8235/baidu/ouro-agen/P2W
git remote set-url origin git@github.com:ootyzzz/P2W.git

提交icode流程：
1. 新建卡片：https://console.cloud.baidu-int.com/devops/icafe/issue/search-agent-1008/show?source=copy-shortcut
2. git add . & git commit -m "search-agent-1008 [Epic] 【P2W】download datasets" & git push origin HEAD:refs/for/main