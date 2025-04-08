import re


class ChapterTree:
    def __init__(self, text, **kwargs):
        self.chapter_text = text
        self.chapter_font = None  # 章节字体样式，目前仅有 {"size": }
        self.next = None  # 同层级的下一个章节节点
        self.child_num = 0
        self.first_child_chapter = None  # 第一个子章节
        self.parent_chapter = None  # 父章节
        self.pattern_index = -1  # 章节样式，对应于编号样式；编号样式定义在settings中
        self.chapter_num = -1  # 章节在样式下的编号  一. 编号为1  1. 编号为1
        self.chapter_cid = None  # 章节编号 id
        self.is_keyinfo = False  # 标示章节是关键信息章节

    def insert(self, node, insert_flag=0):
        if insert_flag == 0:  # 插入到同层级章节
            self.next = node
            node.parent_chapter = self.parent_chapter

        elif insert_flag == 1:  # 插入到当前层级的子章节
            node.parent_chapter = self
            if self.first_child_chapter is None:  # 第一个子章节
                self.first_child_chapter = node
            else:
                p = self.first_child_chapter
                while p.next is not None:  # 末尾的子章节, 找到同层级的最后的章节例如为5.3，现在的层级就是5.4
                    p = p.next
                p.next = node
        # 当前节点的cid等于父节点的cid加上父节点的孩子数量。
        node.parent_chapter.child_num += 1
        node.chapter_cid = node.parent_chapter.chapter_cid + '.{}'.format(node.parent_chapter.child_num)


def insert_chapter_node(last_node, cur_node, chapter_cid, chapter_root, chapter_config):
    """
    将当前章节节点插入到章节目录结构中
    :param chapter_config:
    :param chapter_root:
    :param last_node: 文本中上一处章节节点
    :param cur_node: 当前章节节点
    :param chapter_cid: 当前cid编码
    :return:
    """
    # 如果当前的章节是关键信息章节,插入到最高层级,即根节点的子层级。
    if cur_node.is_keyinfo:
        chapter_root.insert(cur_node, insert_flag=1)
        return True

    # 在循环过程中找到样式相同且章节序号小于当前章节的节点。
    if last_node.is_keyinfo:
        child_node = chapter_root.first_child_chapter
        if cur_node.chapter_font["size"] >= child_node.chapter_font["size"]:
            chapter_root.insert(cur_node, insert_flag=1)
            return True

    # 在循环过程中找到样式相同且章节序号小于当前章节的节点。
    parent_node = last_node
    while parent_node and parent_node.chapter_text != "[CHAPTER_ROOT]":
        if parent_node.pattern_index == cur_node.pattern_index and \
                parent_node.chapter_num < cur_node.chapter_num:
            parent_node.insert(cur_node, insert_flag=0)  # 插入到同层
            return True
        parent_node = parent_node.parent_chapter

    # 没有在父亲节点的遍历过程中找到和自己样式相同的节点，根据字体大小进行判断。
    if cur_node.chapter_font["size"] - last_node.chapter_font["size"] >= chapter_config["chapter_text_size_gap"]:
        last_node.insert(cur_node, insert_flag=0)
        return True

    # 如果样式不相同，字体大小可能相同，且编号为1，2，3，插入到子层级
    if cur_node.chapter_num in [1, 2, 3]:
        cur_layers = len(chapter_cid.split(".")) + 1
        if cur_layers > chapter_config["layers"]:
            return False
        last_node.insert(cur_node, insert_flag=1)
        return True

    return False


# 通过遍历树中所有节点获取节点cid和节点chapter的映射字典。
def preorder(chapter_head):
    cid2chapter = {}
    if chapter_head is None:
        return
    stack = [chapter_head]
    while len(stack) > 0:
        node = stack.pop()
        cid2chapter[node.chapter_cid] = node.chapter_text
        if node.next:
            stack.append(node.next)
        if node.first_child_chapter:
            stack.append(node.first_child_chapter)
    return cid2chapter


def chapter2num(chapter_str):
    """将章节编号对应到其idx，例如：input：（一） return：1；当前只能处理到1～50的编码，待优化 """
    han_dic = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十",
               11: "十一", 12: "十二", 13: "十三", 14: "十四", 15: "十五", 16: "十六", 17: "十七", 18: "十八",
               19: "十九", 20: "二十"}
    num_list = [i for i in range(1, 50)]
    han_list = []
    for num in num_list:
        if num <= 20:
            han_list.append(han_dic[num])
        else:
            han_str = han_dic[int(num / 10)] + "十"
            if num % 10 != 0:
                han_str += han_dic[int(num % 10)]
            han_list.append(han_str)
    num_list = [str(i) for i in num_list]
    res_idx = num_list.index(chapter_str) if chapter_str in num_list else -1
    if res_idx == -1:
        res_idx = han_list.index(chapter_str) if chapter_str in han_list else -1
    # 若能找到对应的章节号，则 +1 表示真正的编号
    res_idx = res_idx + 1 if res_idx != -1 else res_idx
    return res_idx


def num2han(num):
    """将数字转换成中文数字；如果超过万，则分为两部分以节约代码和运行速度"""
    han_dic = {0: "零", 1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七",
               8: "八", 9: "九", 10: "十", 11: "十一", 12: "十二", 13: "十三",
               14: "十四", 15: "十五", 16: "十六", 17: "十七", 18: "十八", 19: "十九", 20: "二十"}
    if num <= 20:
        return han_dic[num]
    num_str = str(num)
    if len(num_str) > 4:
        han_num = tran(num_str[0:-4]) + '万' + tran(num_str[-4:])
    else:
        han_num = tran(num_str[-4:])
    return han_num


def tran(x):
    """转换数字并插入对应单位，单位为‘零’则再插入一个‘零’以方便正则表达式替换"""
    num = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九']
    kin = ['零', '十', '百', '千']
    x = list(reversed(x))
    for i in x:
        x[(x.index(i))] = num[int(i)]
    if len(x) >= 2:
        if x[1] == num[0]:
            x.insert(1, kin[0])
        else:
            x.insert(1, kin[1])
        if len(x) >= 4:
            if x[3] == num[0]:
                x.insert(3, kin[0])
            else:
                x.insert(3, kin[2])
            if len(x) >= 6:
                if x[5] == num[0]:
                    x.insert(5, kin[0])
                else:
                    x.insert(5, kin[3])
    # 进行多余‘零’的删除
    # reversed()函数真是可以用在列表和字符串。
    # 加上 if 语句 防止对不必要的数据进行正则表达式检测
    x = ''.join(x)
    if '零零' in x:
        x = re.sub('零+', '零', x)
    if x.startswith('零'):
        x = list(x)
        x.remove('零')
    x = reversed(x)
    x = ''.join(x)
    return x
