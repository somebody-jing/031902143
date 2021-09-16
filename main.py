import argparse  # 导入argpaerse包
from pypinyin import lazy_pinyin
from pychai import Schema  # 导入pychai库
import re
import time

global count
count = 0

# 定义参数
parser = argparse.ArgumentParser(description='Sensitive word detection')
# 通过对象的add_argument函数来增加参数
parser.add_argument('words', type=str, default='')
parser.add_argument('org', type=str, default='')
parser.add_argument('ans', type=str, default='')
args = parser.parse_args()

"""
实现拆字的功能
"""
def func_chaizi():
    wubi98 = Schema('wubi98')  # 实例化一个 Schema 对象，我们将其命名为 wubi98
    wubi98.run()
    for nameChar in wubi98.charList:
        if nameChar in wubi98.component:
            scheme = wubi98.component[nameChar]
        else:
            tree = wubi98.tree[nameChar]
            componentList = tree.flatten_with_complex(wubi98.complexRootList)
            scheme = sum((wubi98.component[component] for component in componentList), tuple())
        if len(scheme) == 1:
            objectRoot = scheme[0]
            nameRoot = objectRoot.name
            if nameChar in '王土大木工目日口田山禾白月人金言立水火之已子女又幺':
                info = [nameRoot] * 4
            elif nameChar in '一丨丿丶乙':
                info = [nameRoot] * 2 + ['田'] * 2
            else:
                firstStroke = objectRoot.strokeList[0].type
                secondStroke = objectRoot.strokeList[1].type
                if objectRoot.charlen == 2:
                    info = [nameRoot, firstStroke, secondStroke]
                else:
                    lastStroke = objectRoot.strokeList[-1].type
                    info = [nameRoot, firstStroke, secondStroke, lastStroke]
        elif len(scheme) < 4:
            if nameChar in wubi98.component or tree.structure not in 'hz':
                weima = '3'
            elif tree.structure == 'h':
                weima = '1'
            elif tree.structure == 'z':
                weima = '2'
            lastObjectRoot = scheme[-1]
            quma = wubi98.category[lastObjectRoot.strokeList[-1].type]
            shibiema = quma + weima
            info = [objectRoot.name for objectRoot in scheme] + [shibiema]
        elif len(scheme) > 4:
            scheme = scheme[:3] + scheme[-1:]
            info = [objectRoot.name for objectRoot in scheme]
        else:
            info = [objectRoot.name for objectRoot in scheme]
        code = ''.join(wubi98.rootSet[nameRoot] for nameRoot in info)
        wubi98.encoder[nameChar] = code
    return wubi98


def Tosplit(split_word, word):
    # 从每行的开始进行匹配
    if word == ('a' <= word <= 'z') or ('A' < word < 'Z') or word.isdigit() or word == '\n':  # word.isdigit()只有数字
        return '0'
    if word in split_word.tree.keys():
        # Tree.first 和 Tree.second：字按该结构进行二分之后，得到的第一部分（如王）和第二部分（如里）。
        return split_word.tree[word].first.name[0], split_word.tree[word].second.name[0]
    else:
        return '0'

"""
实现敏感词检测
"""
def ToSearchsensitive(sentence_list, rules_list, rules_split_list, pinyinlist, sensitive_words):
    global count
    ans_list=[]
    for line, sentence in enumerate(sentence_list,start=1):
        sensitive_index = []
        for num, rules in enumerate(rules_split_list,start=1):
            for i in re.finditer(rules, sentence, re.I):  # 不区分大小写
                res = 'Line' + str(line) + ':' + ' <' + sensitive_words[num-1] + '> ' + i.group()
                sensitive_index.append((i.span()[0], res))
                count += 1

        location = []
        for index, word in enumerate(sentence):  #对多音字进行处理
            for pinyin in pinyinlist:
                if ''.join(lazy_pinyin(word)) == pinyin[1] and word != pinyin[0]: #当拼音一样而字不一样
                    location.append((index, word))     #找出多音字的下标
                    s1 = list(sentence)
                    s1[index] = pinyin[0]
                    sentence = ''.join(s1)    #进行多音字的替换，替换为敏感词
        for num, rules in enumerate(rules_list):
            for i in re.finditer(rules, sentence, re.I):
                    res = 'Line' + str(line + 1) + ':' + ' <' + sensitive_words[
                        num] + '> ' + i.group()  # group用来提出分组截获的字符串
                    sensitive_index.append((i.span()[0], res))
                    count += 1
        # 将列表中的敏感词按下标位置输出
        for group in sensitive_index:
               ans_list.append(group[1])   #转换成字符串形式输出


    ans = 'Total: ' + str(count) + '\n'
    txt_ans.write(ans)
    for answer in ans_list:
        answer=answer+'\n'
        txt_ans.write(answer)
    '''    
    for answer in sensitive_index:
            print(answer)
            with open(args.answer, 'w') as output_file:
                output_file.write(answer)
                output_file.write('\r\n')
    print(count)
 '''

def creatRules_pinyin(dictionary):  #构造正则表达式的拼音部分规则
    rules = []
    for key in dictionary:     # 键值
            length_key = len(key)
            rules_key = ''
            rules_key_split = ''
            for num, character in enumerate(key,start=1):  #按照词典的键值构造正则表达式，返回数据以及数据下标值(index=1开始)
                if 'a' <= character[0] <= 'z' or 'A' <= character[0] <= 'Z':  # 英文单词
                    if num == length_key:
                        rules_key += '(?:' + character + '|' + character.upper() + ')'
                    else:
                        rules_key += '(?:' + character + '|' + character.upper() + ')[^\\u4e00-\\u9fa5]*'

                else:  # 中文汉字
                    pinyin = ''.join(lazy_pinyin(character))
                    if num == length_key:
                        rules_key += '(?:' + pinyin + '|' + pinyin[
                            0] + '|' + character + ')'
                    else:
                        rules_key += '(?:' + pinyin + '|' + pinyin[0] + '|' + character + ')[^\\u4e00-\\u9fa5]*'
            rules.append(rules_key)
    return rules


def creatRules_split(dictionary):  #构造正则表达式拆字部分规则
    rules = []
    for key in dictionary:  # 键值
            length_key = len(key)
            rules_key = ''
            for num, character in enumerate(key,start=1):  # 每个字
                if 'a' <= character[0] <= 'z' or 'A' <= character[0] <= 'Z':  # 英文单词
                    continue
                else:  # 中文汉字
                    character_split = ''
                    for pianpang in dictionary[key][num-1]:
                        if pianpang == '0':
                            continue
                        else:
                            character_split += pianpang     # pianpang= '氵去'
                    if num == length_key:
                        rules_key += '(?:' + character_split + ')'
                    else:
                        rules_key += '(?:' + character_split + ')[^\\u4e00-\\u9fa5]*'
            if rules_key != '':
                rules.append(rules_key)
    return rules

def Creatdictionary(sensitive_words, text):
    pinyinlist = []
    for word in sensitive_words:
        pinyin_word_list = []
        for character in word:
            if 'a' <= character <= 'z' or 'A' <= character <= 'Z':
                break
            else:
                pinyinlist.append((character, ''.join(lazy_pinyin(character))))  #构造拼音列表,列表元素为元组[('法','fa'),('轮’,'lun')]

    dictionary = {}
    split = func_chaizi()
    for word in sensitive_words:
        if 'a' <= word[0] <= 'z' or 'A' <= word[0] <= 'Z':  #敏感词为英文
            Word_dictionary = {word: None}    #当敏感词是英文时，key=word，值为None
            dictionary.update(Word_dictionary)  #载入词典
            continue
        split_word_list = []
        for character in word:         #进行拆字
            split_word_list.append(Tosplit(split, character))
        split_word_tuple = tuple(split_word_list)  #tuplr函数将列表转换为元组
        Word_dictionary = {word: split_word_tuple}  #若是中文，此时key=word,值为二分拆字的结果
        dictionary.update(Word_dictionary)      #载入词典
    ToSearchsensitive(text, creatRules_pinyin(dictionary), creatRules_split(dictionary), pinyinlist, sensitive_words)  #开始敏感词检测

if __name__ == '__main__':
    #读取文件
    txt_word = open(args.words, 'r', encoding='utf-8')
    txt_org = open(args.org, 'r', encoding='utf-8')  #进行IO检验
    try:
        org_file = open('org.txt', encoding='utf-8')
    except IOError:
        print("Not Found")
    else:
        print("Successfully Load")
    txt_ans = open(args.ans, 'w', encoding='utf-8')
    read_ans = txt_word.read()
    sensitive_words = read_ans.split('\n')  #通过换行符进行切块
    text = txt_org.readlines()
    Creatdictionary(sensitive_words, text)  #构造敏感词词典
