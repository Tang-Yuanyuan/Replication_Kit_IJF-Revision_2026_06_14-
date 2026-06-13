# 项目规则

## 检查点工作流（必须遵守）

**每次修改文件之前**，必须先执行：
```
git add -A && git commit -m "checkpoint: <简短描述当前状态>"
```

**当用户说"回到上一步"**（或类似表达，如"撤销"、"undo"、"上一步"），立即执行：
```
git reset --hard HEAD~1
```
然后告知用户已回滚到哪个 commit。

**规则说明：**
- 不需要用户每次提醒，这是默认工作流
- commit message 用中文，格式：`checkpoint: <状态描述>`
- `reset --hard` 会丢弃未提交内容，因此检查点 commit 必须在修改前完成
