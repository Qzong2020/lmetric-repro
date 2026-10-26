# 阶段一：源码核对与协议设计

已完成源码调用链核对，设计见 docs/kv_event_journal_design.md。关键发现：cache_full_blocks 是映射元数据登记，可早于设备执行；free 不等于 evict；全局删除没有请求上下文；async 可在输出 update 登记更多映射。采用独立有界 journal、全局映射 token proof、请求完整占用表、连续 mutation_seq 和 idle reset 控制 envelope。已有源码完整副本、tracked/untracked/子模块哈希及历史验收文件哈希已保存；NPU空闲。进入阶段二，先编写失败单元测试再加生产钩子。
