# 中文分词

# 正向最大匹配
class MM(object):
    def __init__(self, window_size=3):
        self.window_size = window_size

    def cut(self, text):
        result = []
        index = 0 
        text_length =len(text)
        dic = ['研究', '研究生', '生命', '命', '的', '起源']
        while index < text_length:
            for size in range(index + self.window_size, index, -1):
                piece = text[index:size]
                if piece in dic:
                    result.append(piece)
                    index = size - 1
                    break 
            index += 1
        print(result)

# 逆向最大匹配
class RMM(object):
    def __init__(self, window_size=3):
        self.window_size = window_size

    def cut(self, text):
        result = []
        text_length = len(text)
        index = text_length
        dic = ['研究', '研究生', '生命', '命', '的', '起源']
        while index > 0:
            for size in range(index-self.window_size, index, 1):
                piece = text[size:index]
                if piece in dic:
                    index = size + 1
                    break 
            result.append(piece)
            index -= 1
        result.reverse()
        print(result)



if __name__ == "__main__":
    text = '研究生命的起源'
    RMM(3).cut(text)                   


    