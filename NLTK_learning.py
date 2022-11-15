import matplotlib
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

# 用stopwords库帮助去除I，You，The等无实际意义的词，提高词频统计精度
# 去除换行符，连字符等无实际意义的标点（这个步骤与去除 I You The的清洗可以合并进行）
stop_words = stopwords.words('english')
for w in ['!', ',', '.', '?', '-s', '-ly', '</s>', 's', '``', '\'', '\"', '[', ']', 'ma', '-']:
    stop_words.append(w)

# 读取文件
def read_file(path):
    file = open(path)
    text = file.read()
    file.close()
    return text

# 文本分词
def generate_corpus(text):
    sentences = nltk.sent_tokenize(text)
    corpus = nltk.word_tokenize(str(sentences))
    return corpus

# 词性标注
def word_tagging(corpus):
    tagged_words = nltk.pos_tag(corpus)
    return tagged_words

# 去除停用词
def generate_filtered_words(corpus):
    filtered_words = [word for word in corpus if word.lower() not in stop_words]
    return filtered_words

# 去除前后缀
def remove_suffix_prefix(filtered_words):
    for i in range(len(filtered_words)):
        filtered_words[i] = filtered_words[i].strip()
        filtered_words[i] = filtered_words[i].strip('\'')
        filtered_words[i] = filtered_words[i].strip('\n')
        filtered_words[i] = filtered_words[i].strip('\\n')
        filtered_words[i] = filtered_words[i].strip('\\')
        filtered_words[i] = filtered_words[i].strip('/')
    return filtered_words

# 按动词、名词、形容词、副词简单分为四类
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatized_words(tagged_words):  # 词形还原
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word_tag in tagged_words:
        wordnet_pos = get_wordnet_pos(word_tag[1]) or wordnet.NOUN
        lemmas.append(lemmatizer.lemmatize(word_tag[0], pos=wordnet_pos))
    return lemmas

    # 生成词云
def WordCloudGeneration(src, freq_dist):
    Word_C = WordCloud(
        background_color="white",  # 设置背景为白色，默认为黑色
        width=1440,  # 设置图片的宽度
        height=900,  # 设置图片的高度
        margin=10  # 设置图片的边缘
    ).generate(str(freq_dist.most_common(30)))
    # 绘制图片
    matplotlib.pyplot.imshow(Word_C)
    # 消除坐标轴
    matplotlib.pyplot.axis("off")
    # 展示图片
    matplotlib.pyplot.show()
    # 保存图片
    Word_C.to_file(src + '_wordcloud.jpg')

if __name__ == "__main__":

    path = "C:/Users/Windows/Desktop/Jane Eyre.txt"

    # 读取文件
    text = read_file(path)

    # 文本分词
    corpus = generate_corpus(text)

    # 去除停用词
    filtered_words = generate_filtered_words(corpus)

    # 去除前后缀
    filtered_words = remove_suffix_prefix(filtered_words)

    # 二次过滤
    filtered_words = generate_filtered_words(filtered_words)

    # 词性标注
    tagged_words = word_tagging(filtered_words)

    # 词形还原
    lemmas = lemmatized_words(tagged_words)

    # 词频统计
    freq_dist = nltk.FreqDist(lemmas)

    # 生成词云
    WordCloudGeneration('jane_eyre', freq_dist)