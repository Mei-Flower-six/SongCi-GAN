import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import json
import os
from collections import Counter
import random
import time
import gc

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 词表构建及序列预处理
class DataPreprocess:
    def __init__(self, data_dir, min_freq=2, val_ratio=0.15):
        """
        初始化数据预处理类
        :param data_dir: 数据目录
        :param min_freq: 最小词频，低于此频率的词将被过滤
        :param val_ratio: 验证集比例
        """
        self.data_dir = data_dir
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ['<START>', '<END>', '<PAD>', '<UNK>']  # 特殊标记
        self.max_seq_length = 0
        self.min_freq = min_freq
        self.val_ratio = val_ratio
        self.keyword_vocab = set()
        self.rhythm_vocab = set()
        self.all_songs = []
        self.all_keywords = []
        self.all_authors = []
        self.all_rhythms = []
        self.train_songs = []
        self.val_songs = []

        # 主题意象库 - 用于指导诗词生成的语义方向
        self.theme_images = {
            '春': ['柳岸', '杏雨', '梨云', '燕归', '莺啼', '草长', '桃花源', '春风', '花开'],
            '秋': ['枫红', '雁阵', '菊香', '桂影', '霜天', '蝉鸣', '乡思浓', '落叶', '秋月'],
            '月': ['明月', '西楼', '桂影', '寒江', '清辉', '月钩', '月影斜', '玉盘', '银辉'],
            '乡思': ['孤舟', '故园', '归雁', '长亭', '天涯', '旧梦', '笛声残', '故里', '乡愁'],
            '雪': ['雪花', '玉梅', '琼枝', '霜天', '寒梅'],
            '相思': ['红豆', '相思泪', '忆江南', '长相思'],
            '雨': ['细雨', '雷雨', '烟雨', '雨声', '滴阶', '霖铃', '霏微', '雾雨']
        }

        # 词牌格律定义
        self.rhythm_patterns = {
            '浣溪沙': [7, 7, 7, 7, 7, 7],  # 6句，每句7字
            '蝶恋花': [7, 4, 5, 7, 7, 4, 5, 7],  # 8句，特定字数组合
            '临江仙': [6, 6, 7, 5, 5, 6, 6, 7, 5, 5],  # 10句，特定字数组合
            '西江月': [6, 6, 7, 6, 6, 6, 7, 6],  # 8句，特定字数组合
            '鹧鸪天': [7, 7, 7, 7, 7, 7, 3, 3, 7]  # 9句，特定字数组合
        }

    def load_data(self, sample_data=None):
        """
        加载全部宋词数据，移除数据量限制
        :param sample_data: 可选的样本数据，用于测试
        :return: 训练集、验证集、关键词、作者和词牌列表
        """
        print("开始加载宋词数据...")
        file_count = 0

        if sample_data:
            print(f"使用提供的样本数据 ({len(sample_data)} 条)")
            # 处理样本数据
            for i, item in enumerate(sample_data):
                paragraphs = item.get('paragraphs', [])
                if not paragraphs:
                    continue

                song_text = ' '.join(paragraphs).strip()
                self.all_songs.append(song_text)
                author = item.get('author', '佚名')
                rhythmic = item.get('rhythmic', '未知词牌')
                self.all_authors.append(author)
                self.all_rhythms.append(rhythmic)
                self.rhythm_vocab.add(rhythmic)

                full_text = ''.join(paragraphs)
                keywords = self.extract_keywords(full_text)
                self.all_keywords.append(keywords)
                self.keyword_vocab.update(keywords)
            file_count = len(sample_data)
        else:
            # 从文件加载完整数据
            target_dir = os.path.join(self.data_dir, '宋词')
            if not os.path.exists(target_dir):
                print(f"错误: 数据目录 {target_dir} 不存在")
                return [], [], []

            print(f"从目录 {target_dir} 加载全部数据...")
            files = [f for f in os.listdir(target_dir) if f.endswith('.json') and "author.song.json" not in f]
            if not files:
                print(f"错误: 目录 {target_dir} 中没有找到JSON文件")
                return [], [], []

            for file in files:
                file_path = os.path.join(target_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if not isinstance(data, list):
                            print(f"警告: 文件 {file} 不是列表格式，跳过")
                            continue

                        print(f"处理文件 {file} ({len(data)} 条记录)")
                        for item in data:
                            paragraphs = item.get('paragraphs', [])
                            if not paragraphs:
                                continue

                            song_text = ' '.join(paragraphs).strip()
                            self.all_songs.append(song_text)

                            author = item.get('author', '佚名')
                            self.all_authors.append(author)

                            rhythmic = item.get('rhythmic', '未知词牌')
                            self.all_rhythms.append(rhythmic)
                            self.rhythm_vocab.add(rhythmic)

                            full_text = ''.join(paragraphs)
                            keywords = self.extract_keywords(full_text)
                            self.all_keywords.append(keywords)
                            self.keyword_vocab.update(keywords)
                    file_count += 1
                except Exception as e:
                    print(f"加载文件 {file_path} 时出错: {e}")

        self.all_songs = [song for song in self.all_songs if song]  # 移除空字符串
        print(f"成功加载 {len(self.all_songs)} 首宋词 (来自 {file_count} 个文件)")

        # 划分训练集和验证集
        val_size = int(len(self.all_songs) * self.val_ratio)
        indices = list(range(len(self.all_songs)))
        random.shuffle(indices)

        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        self.train_songs = [self.all_songs[i] for i in train_indices]
        self.val_songs = [self.all_songs[i] for i in val_indices]
        self.train_keywords = [self.all_keywords[i] for i in train_indices]
        self.val_keywords = [self.all_keywords[i] for i in val_indices]

        return self.train_songs, self.val_songs, self.all_keywords, self.all_authors, self.all_rhythms

    def extract_keywords(self, text):
        """
        从文本中提取关键词，过滤常见停用词
        :param text: 输入文本
        :return: 关键词列表
        """
        stopwords = {'，', '。', '？', '！', '；', '：', '、', ' ', '\n', '也', '的', '了', '在', '是', '和', '与'}
        keywords = [char for char in text if char not in stopwords]
        return list(set(keywords))

    def build_vocab(self, texts):
        """
        构建词表，将文本中的字符映射到索引
        :param texts: 文本列表
        :return: 词汇表
        """
        all_chars = []
        for text in texts:
            all_chars.extend(list(text))
        char_count = Counter(all_chars)
        vocab = self.special_tokens + [char for char, cnt in char_count.items() if cnt >= self.min_freq]
        self.word2idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx2word = {idx: char for char, idx in self.word2idx.items()}
        return vocab

    def process_sequences(self, texts):
        """
        将文本转换为模型输入的序列，添加开始和结束标记，并进行填充
        :param texts: 文本列表
        :return: 处理后的序列列表
        """
        sequences = []
        self.max_seq_length = 0
        # 将文本转换为索引序列，并添加开始和结束标记
        for text in texts:
            seq = [self.word2idx.get(char, self.word2idx['<UNK>']) for char in text]
            seq = [self.word2idx['<START>']] + seq + [self.word2idx['<END>']]
            sequences.append(seq)
            self.max_seq_length = max(self.max_seq_length, len(seq))
        # 序列填充
        padded_sequences = []
        max_len = min(128, self.max_seq_length)  # 限制最大序列长度
        for seq in sequences:
            if len(seq) > max_len:
                seq = seq[:max_len - 1] + [self.word2idx['<END>']]
            padded = seq + [self.word2idx['<PAD>']] * (max_len - len(seq))
            padded_sequences.append(padded)
        self.max_seq_length = max_len
        return padded_sequences

# Seq2Seq生成器
class Seq2SeqGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.2, word2idx=None):
        """
        初始化序列到序列生成器
        :param vocab_size: 词汇表大小
        :param embedding_dim: 嵌入维度
        :param hidden_dim: 隐藏层维度
        :param num_layers: GRU层数
        :param dropout: Dropout比率
        :param word2idx: 词到索引的映射
        """
        super(Seq2SeqGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 词嵌入层
        self.dropout = nn.Dropout(dropout)  # Dropout层，防止过拟合

        # 存储word2idx，如果提供的话
        self.word2idx = word2idx

        # 编码器 - 单向GRU
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        # 解码器 - 单向GRU，使用编码器的隐藏状态作为初始状态
        self.decoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)  # 全连接层，用于生成下一个词的概率分布
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 如果没有提供word2idx，使用默认的特殊标记索引
        self.start_idx = 0  # 默认<START>在位置0
        self.pad_idx = 2  # 默认<PAD>在位置2
        if word2idx is not None:
            self.start_idx = word2idx.get('<START>', 0)
            self.pad_idx = word2idx.get('<PAD>', 2)

    def forward(self, x, hidden=None):
        """
        模型前向传播
        :param x: 输入序列
        :param hidden: 初始隐藏状态
        :return: 输出序列和最终隐藏状态
        """
        batch_size, seq_len = x.size()
        embedded = self.dropout(self.embedding(x))

        # 编码器前向传播
        encoder_output, hidden = self.encoder(embedded)

        # 解码器初始输入：<START>标记的嵌入向量
        start_token = torch.tensor([self.start_idx] * batch_size, dtype=torch.long, device=x.device)
        decoder_input = self.embedding(start_token).unsqueeze(1)  # [batch_size, 1, embedding_dim]

        # 存储解码器输出
        decoder_outputs = []

        # 逐时间步解码
        for t in range(seq_len):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            output = self.fc(decoder_output)
            decoder_outputs.append(output)

            # 教师强制：使用真实标签作为下一个输入
            if t < seq_len - 1:
                decoder_input = embedded[:, t + 1:t + 2, :]  # 使用下一个真实标记的嵌入向量

        # 合并所有时间步的输出
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs, hidden

    def init_hidden(self, batch_size, device):
        """
        初始化隐藏状态
        :param batch_size: 批量大小
        :param device: 设备类型
        :return: 初始化的隐藏状态
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

# 轻量级判别器
class LightweightDiscriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=192, num_layers=1, dropout=0.2):
        """
        初始化轻量级判别器
        :param vocab_size: 词汇表大小
        :param embedding_dim: 嵌入维度
        :param hidden_dim: 隐藏层维度
        :param num_layers: GRU层数
        :param dropout: Dropout比率
        """
        super(LightweightDiscriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 使用单向GRU
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )

        self.fc = nn.Linear(hidden_dim, 1)  # 输出层，用于判断序列是否为真实的
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        判别器前向传播
        :param x: 输入序列
        :return: 判断结果的概率
        """
        embedded = self.dropout(self.embedding(x))
        output, _ = self.gru(embedded)
        output = self.dropout(output[:, -1, :])
        validity = self.fc(output)
        return torch.sigmoid(validity)

# 训练类
class EnhancedGANTrainer:
    def __init__(self, generator, discriminator, word2idx, idx2word, device, rhythm_patterns):
        """
        初始化增强型GAN训练器
        :param generator: 生成器模型
        :param discriminator: 判别器模型
        :param word2idx: 词到索引的映射
        :param idx2word: 索引到词的映射
        :param device: 设备类型
        :param rhythm_patterns: 词牌格律模式
        """
        self.generator = generator
        self.discriminator = discriminator
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.device = device
        self.rhythm_patterns = rhythm_patterns  # 词牌格律

        # 优化器（进一步降低学习率）
        self.g_optimizer = optim.Adam(generator.parameters(), lr=0.0004)
        self.d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0004)

        # 损失函数
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=word2idx['<PAD>'])  # 交叉熵损失
        self.bce_loss = nn.BCELoss()  # 二元交叉熵损失（用于GAN）
        self.repeat_loss_weight = 0.3  # 进一步增加重复字损失权重
        self.rhythm_loss_weight = 0.1  # 新增格律损失权重

        # 常见字符，用于控制重复率
        self.common_chars = ['春', '秋', '月', '花', '风', '雨', '山', '水',
                             '江', '云', '天', '人', '心', '情', '思', '愁',
                             '红', '绿', '黄', '白', '青', '香', '影', '梦',
                             '明', '乡', '思', '归', '故']
        self.end_token = word2idx.get('<END>', None)
        self.punctuation = {'。', '？', '！', '；', '，', '：', '、'}

        # 词牌名列表（使用有格律定义的词牌）
        self.rhythm_names = list(rhythm_patterns.keys())

        # 主题相关意象库（转换为字符集合）
        self.theme_char_set = set()
        for theme_images in self._get_theme_images().values():
            for image in theme_images:
                self.theme_char_set.update(image)

        # 主题相关意象库（保留原始格式）
        self.theme_images = {
            '春': ['柳岸', '杏雨', '梨云', '燕归', '莺啼', '草长', '桃花源', '春风', '花开'],
            '秋': ['枫红', '雁阵', '菊香', '桂影', '霜天', '蝉鸣', '乡思浓', '落叶', '秋月'],
            '月': ['明月', '西楼', '桂影', '寒江', '清辉', '月钩', '月影斜', '玉盘', '银辉'],
            '乡思': ['孤舟', '故园', '归雁', '长亭', '天涯', '旧梦', '笛声残', '故里', '乡愁'],
            '雪': ['雪花', '玉梅', '琼枝', '霜天', '寒梅'],
            '相思': ['红豆', '相思泪', '忆江南', '长相思'],
            '雨': ['细雨', '雷雨', '烟雨', '雨声', '滴阶', '霖铃', '霏微', '雾雨']
        }

        # 常见宋词用字（扩展版）
        self.fallback_chars = ['风', '花', '雪', '月', '春', '秋', '山', '水',
                               '江', '云', '天', '人', '情', '思', '愁', '楼',
                               '明', '乡', '归', '故', '梦', '影', '香', '雨',
                               '叶', '枝', '飞', '来', '去', '看', '见', '闻', '听',
                               '东', '南', '西', '北', '前', '后', '左', '右', '中',
                               '新', '旧', '远', '近', '高', '低', '早', '晚', '朝', '暮',
                               '楼', '台', '亭', '阁', '舟', '桥', '路', '径', '村']

        # 意象组合库（增强版，包含更多主题相关组合）
        self.image_combinations = {
            '春': [('柳岸', '莺啼'), ('杏雨', '梨云'), ('春风', '花开'), ('燕归', '草长'), ('桃花源', '流水')],
            '秋': [('枫红', '雁阵'), ('菊香', '桂影'), ('霜天', '落叶'), ('蝉鸣', '乡思'), ('秋月', '寒江')],
            '月': [('明月', '西楼'), ('桂影', '寒江'), ('清辉', '玉盘'), ('月钩', '月影'), ('月光', '如水')],
            '雨': [('细雨', '春风'), ('雷雨', '秋声'), ('烟雨', '江南'), ('雨声', '滴阶'), ('霖铃', '愁绪')],
            '花': [('花开', '花落'), ('花香', '花影'), ('红花', '绿叶'), ('花前', '月下'), ('花海', '蝶舞')],
            '雪': [('雪花', '玉梅'), ('琼枝', '霜天'), ('寒梅', '映雪'), ('雪夜', '归人'), ('风雪', '孤舟')]
        }

    def _get_theme_images(self):
        """获取主题意象（返回字符集合）"""
        theme_images = {
            '春': ['柳岸', '杏雨', '梨云', '燕归', '莺啼', '草长', '桃花源', '春风', '花开'],
            '秋': ['枫红', '雁阵', '菊香', '桂影', '霜天', '蝉鸣', '乡思浓', '落叶', '秋月'],
            '月': ['明月', '西楼', '桂影', '寒江', '清辉', '月钩', '月影斜', '玉盘', '银辉'],
            '乡思': ['孤舟', '故园', '归雁', '长亭', '天涯', '旧梦', '笛声残', '故里', '乡愁'],
            '雪': ['雪花', '玉梅', '琼枝', '霜天', '寒梅'],
            '相思': ['红豆', '相思泪', '忆江南', '长相思'],
            '雨': ['细雨', '雷雨', '烟雨', '雨声', '滴阶', '霖铃', '霏微', '雾雨']
        }
        return theme_images

    def train_gan(self, train_dataloader, val_dataloader, epochs=30, d_steps=3):
        """
        训练GAN模型
        :param train_dataloader: 训练数据加载器
        :param val_dataloader: 验证数据加载器
        :param epochs: 训练轮数
        :param d_steps: 每轮训练中判别器的更新次数
        :return: 最佳验证损失
        """
        # 标签
        valid = torch.ones(train_dataloader.batch_size, 1, device=self.device)
        fake = torch.zeros(train_dataloader.batch_size, 1, device=self.device)

        best_val_loss = float('inf')
        no_improve = 0

        print("开始训练GAN模型...")
        for epoch in range(epochs):
            # 训练判别器
            self.discriminator.train()
            d_losses = []

            max_d_steps = min(d_steps, len(train_dataloader))
            for step, batch in enumerate(train_dataloader):
                if step >= max_d_steps:
                    break

                real_sequences = batch[0].to(self.device)
                real_validity = self.discriminator(real_sequences)
                d_real_loss = self.bce_loss(real_validity, valid)

                fake_sequences = self._generate_fake_sequences(real_sequences.shape[0])
                fake_validity = self.discriminator(fake_sequences)
                d_fake_loss = self.bce_loss(fake_validity, fake)

                d_loss = (d_real_loss + d_fake_loss) / 2
                d_losses.append(d_loss.item())

                self.d_optimizer.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.d_optimizer.step()

            # 训练生成器
            self.generator.train()
            g_losses = []

            max_g_steps = min(len(train_dataloader), 10)
            for step, batch in enumerate(train_dataloader):
                if step >= max_g_steps:
                    break

                sequences = batch[0].to(self.device)
                inputs = sequences[:, :-1]
                targets = sequences[:, 1:]

                gen_outputs, _ = self.generator(inputs)
                g_ce_loss = self.ce_loss(gen_outputs.reshape(-1, gen_outputs.shape[-1]), targets.reshape(-1))

                gen_sequences = torch.argmax(gen_outputs, dim=2)
                gen_validity = self.discriminator(gen_sequences)
                g_gan_loss = self.bce_loss(gen_validity, valid)

                # 计算主题一致性损失
                theme_loss = self._calculate_theme_loss(gen_sequences)

                # 计算重复字损失（增强版）
                repeat_loss = self._calculate_repeat_loss(gen_sequences)

                # 计算格律损失（新增）
                rhythm_loss = self._calculate_rhythm_loss(gen_sequences)

                # 多任务损失组合
                g_loss = g_ce_loss + 0.05 * g_gan_loss + 0.15 * theme_loss + self.repeat_loss_weight * repeat_loss + self.rhythm_loss_weight * rhythm_loss
                g_losses.append(g_loss.item())

                self.g_optimizer.zero_grad()
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=0.8)
                self.g_optimizer.step()

            # 验证
            val_loss = self.validate(val_dataloader)
            print(
                f"Epoch {epoch + 1}/{epochs}, D Loss: {np.mean(d_losses):.4f}, G Loss: {np.mean(g_losses):.4f}, Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'g_optimizer': self.g_optimizer.state_dict(),
                    'd_optimizer': self.d_optimizer.state_dict(),
                    'loss': val_loss,
                }, 'best_gan_model.pth')
                print(f"保存最佳模型，验证损失: {val_loss:.4f}")
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 5:
                    print(f"验证损失连续 {no_improve} 轮未提升，提前停止训练")
                    break

        print(f"训练完成，最佳验证损失: {best_val_loss:.4f}")
        return best_val_loss

    def _generate_fake_sequences(self, batch_size, max_len=100):
        """生成假序列用于训练判别器"""
        self.generator.eval()
        device = self.device
        start_token = self.word2idx.get('<START>', 0)
        fake_sequences = []

        with torch.no_grad():
            input_seq = torch.tensor([[start_token]] * batch_size, dtype=torch.long, device=device)
            hidden = self.generator.init_hidden(batch_size, device)

            for i in range(max_len):
                output, hidden = self.generator(input_seq, hidden)
                next_token = torch.argmax(output[:, -1, :], dim=1).unsqueeze(1)
                input_seq = torch.cat([input_seq, next_token], dim=1)

                if (next_token.squeeze() == self.end_token).all():
                    break

            fake_sequences = input_seq

        return fake_sequences

    def validate(self, dataloader):
        """验证生成器性能"""
        self.generator.eval()
        total_loss = 0
        ce_loss = nn.CrossEntropyLoss(ignore_index=self.word2idx['<PAD>'])
        total_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                sequences = batch[0].to(self.device)
                inputs = sequences[:, :-1]
                targets = sequences[:, 1:]

                outputs, _ = self.generator(inputs)
                outputs_flat = outputs.reshape(-1, outputs.shape[-1])
                targets_flat = targets.reshape(-1)

                loss = ce_loss(outputs_flat, targets_flat)
                total_loss += loss.item()
                total_batches += 1

                if total_batches >= 5:
                    break

        return total_loss / total_batches if total_batches > 0 else float('inf')

    def _calculate_theme_loss(self, sequences):
        """计算主题一致性损失"""
        theme_images = self._get_theme_images()
        theme_char_set = set()
        for images in theme_images.values():
            for image in images:
                theme_char_set.update(image)

        loss = torch.tensor(0.0, device=self.device)
        for seq in sequences:
            chars = [self.idx2word.get(idx.item(), '') for idx in seq if
                     idx != self.word2idx['<PAD>'] and idx != self.word2idx['<END>']]
            theme_chars = set(chars) & theme_char_set
            loss -= len(theme_chars) / max(1, len(chars))
        return loss / sequences.size(0)

    def _calculate_repeat_loss(self, sequences):
        """计算重复字损失"""
        loss = torch.tensor(0.0, device=self.device)
        for seq in sequences:
            chars = [self.idx2word.get(idx.item(), '') for idx in seq if
                     idx != self.word2idx['<PAD>'] and idx != self.word2idx['<END>']]
            char_count = Counter(chars)
            # 关键词允许出现2次，其他字允许出现2次
            repeat_ratio = sum(
                cnt for char, cnt in char_count.items() if cnt > (2 if char in self.common_chars else 2)) / max(1,
                                                                                                                len(chars))
            loss += repeat_ratio
        return loss / sequences.size(0)

    def _calculate_rhythm_loss(self, sequences):
        """计算格律损失（新增）"""
        loss = torch.tensor(0.0, device=self.device)
        for seq in sequences:
            chars = [self.idx2word.get(idx.item(), '') for idx in seq if
                     idx != self.word2idx['<PAD>'] and idx != self.word2idx['<END>']]
            # 简单格律损失：惩罚过长或过短的句子
            if len(chars) < 10 or len(chars) > 100:  # 设定合理长度范围
                loss += abs(len(chars) - 50) / 100  # 归一化损失
        return loss / sequences.size(0)

    def generate_poem(self, keyword, max_len=100, temperature=0.75):
        """
        根据关键词生成宋词
        :param keyword: 关键词
        :param max_len: 最大生成长度
        :param temperature: 控制生成的随机性，值越小越确定性，值越大越随机
        :return: 生成的诗词文本
        """
        self.generator.eval()

        # 处理关键词
        valid_keyword = keyword
        if not keyword:
            valid_keyword = random.choice(self.common_chars)
        else:
            valid_chars = [char for char in keyword if char in self.word2idx or char in self.common_chars]
            if not valid_chars:
                valid_keyword = random.choice(self.common_chars)
            else:
                valid_keyword = ''.join(valid_chars)

        start_token = self.word2idx.get('<START>', 0)
        input_seq = torch.tensor([[start_token]], dtype=torch.long, device=self.device)
        hidden = self.generator.init_hidden(1, self.device)

        generated_chars = []
        keyword_boost = 1.5  # 进一步降低关键词增强权重
        end_token = self.end_token
        end_counter = 0
        has_generated_content = False
        punctuation_count = 0
        last_chars = []
        used_images = set()
        theme = self._infer_theme(valid_keyword)

        # 选择词牌并获取格律（严格模式）
        rhythm_name = random.choice(self.rhythm_names)
        rhythm = self.rhythm_patterns.get(rhythm_name, [7, 7, 7, 7, 7, 7])
        sentence_counts = [0]
        current_sentence = 0
        max_sentences = len(rhythm)
        prev_line = []  # 记录上一句，用于语义衔接
        used_keyword = 0  # 记录关键词使用次数

        with torch.no_grad():
            for i in range(max_len):
                output, hidden = self.generator(input_seq, hidden)
                logits = output[0, -1, :] / temperature  # 应用temperature参数，控制随机性

                # 增强关键词和主题相关字（进一步降低关键词权重）
                for char in valid_keyword:
                    if char in self.word2idx and used_keyword < 2:  # 限制关键词使用次数
                        logits[self.word2idx[char]] += keyword_boost

                for img in self.theme_images.get(theme, []):
                    for char in img:
                        if char in self.word2idx and char not in used_images:
                            logits[self.word2idx[char]] += 1.5
                            used_images.add(char)

                # 避免重复字符连续出现（严格版）
                if len(last_chars) > 1:
                    for idx in range(len(logits)):
                        char = self.idx2word.get(idx, '')
                        if char in last_chars[-2:]:
                            logits[idx] -= 1.0  # 增加重复字惩罚

                # 语义衔接：根据上一句生成下一句
                if prev_line and len(generated_chars) > sum(rhythm) * 0.5:
                    last_char = prev_line[-1] if prev_line else ''
                    for idx in range(len(logits)):
                        char = self.idx2word.get(idx, '')
                        if char and last_char and self._is_semantically_related(char, last_char):
                            logits[idx] += 0.7

                # Top-k采样（增加k值以提高多样性）
                top_k = 60
                top_logits, top_indices = torch.topk(logits, top_k)
                top_probs = torch.softmax(top_logits, dim=-1)

                next_token_idx = torch.multinomial(top_probs, 1)
                next_token = top_indices[next_token_idx]
                char = self.idx2word.get(next_token.item(), random.choice(self.fallback_chars))
                last_chars.append(char)
                if len(last_chars) > 4:
                    last_chars.pop(0)

                # 严格格律约束（不允许超过目标长度）
                if current_sentence < max_sentences:
                    target_length = rhythm[current_sentence]
                    if sentence_counts[current_sentence] >= target_length:  # 不允许超过目标长度
                        char = '，' if current_sentence % 2 == 0 else '。'
                        current_sentence += 1
                        if current_sentence < max_sentences:
                            sentence_counts.append(0)
                            prev_line = generated_chars.copy()  # 保存上一句
                            used_keyword += 1 if valid_keyword in ''.join(prev_line) else 0
                        else:
                            end_counter += 1
                    else:
                        generated_chars.append(char)
                        sentence_counts[current_sentence] += 1
                else:
                    if char in self.punctuation:
                        generated_chars.append(char)
                        punctuation_count += 1
                        end_counter += 1

                # 结束条件
                if next_token.item() == end_token:
                    end_counter += 1
                    if (has_generated_content and end_counter >= 2 and
                            punctuation_count >= max_sentences):
                        break
                else:
                    has_generated_content = True

                input_seq = torch.cat([input_seq, next_token.unsqueeze(0)], dim=1)

        poem_text = self.postprocess_poem(generated_chars, valid_keyword, theme, rhythm_name, rhythm)
        return poem_text

    def _infer_theme(self, keyword):
        """推断关键词主题（增强版）"""
        theme_dict = {
            '春': {'春', '柳', '花', '燕', '莺', '杏', '梨'},
            '秋': {'秋', '枫', '雁', '菊', '桂', '霜', '蝉'},
            '月': {'月', '明', '辉', '影', '楼', '盘', '钩'},
            '雨': {'雨', '霖', '霏', '雾', '滴', '瀑', '雷'},
            '花': {'花', '红', '香', '瓣', '枝', '丛', '海'},
            '乡思': {'乡', '思', '归', '故', '园', '亭', '涯'},
            '雪': {'雪', '梅', '琼', '霜', '寒', '玉'}
        }
        keyword_chars = set(keyword)
        theme_scores = {t: len(keyword_chars & theme_chars) for t, theme_chars in theme_dict.items()}
        if not theme_scores:
            return '春'
        return max(theme_scores.items(), key=lambda x: x[1])[0]

    def _is_semantically_related(self, char1, char2):
        """判断两个字符是否语义相关（增强版）"""
        related_pairs = [
            ('风', '花'), ('花', '月'), ('春', '花'), ('秋', '月'),
            ('云', '天'), ('山', '水'), ('江', '水'), ('乡', '思'),
            ('柳', '莺'), ('枫', '雁'), ('梅', '雪'), ('桂', '影'),
            ('桃', '源'), ('梨', '云'), ('杏', '雨'), ('菊', '香'),
            ('雨', '滴'), ('雪', '花'), ('月', '明'), ('风', '雨')
        ]
        return (char1, char2) in related_pairs or (char2, char1) in related_pairs

    def postprocess_poem(self, chars, keyword, theme, rhythm_name, rhythm):
        """增强后处理逻辑，提升语义连贯性"""
        # 过滤特殊标记并处理生僻字
        filtered_chars = []
        for c in chars:
            if c in ['<END>', '<PAD>']:
                continue
            if c in self.common_chars or any(c in img for img in self.theme_images.values()):
                filtered_chars.append(c)
            else:
                filtered_chars.append(random.choice(self.fallback_chars))

        # 确保有足够字符
        while len(filtered_chars) < sum(rhythm) * 0.8:
            filtered_chars.append(random.choice(self.fallback_chars))

        # 按格律分割句子（严格版）
        lines = []
        char_index = 0
        for i, length in enumerate(rhythm):
            line_chars = []
            # 收集当前句子的字符
            while char_index < len(filtered_chars) and len(line_chars) < length:
                line_chars.append(filtered_chars[char_index])
                char_index += 1
            # 填充到准确长度（使用常见字符补全）
            while len(line_chars) < length:
                line_chars.append(random.choice(self.fallback_chars))
            lines.append(''.join(line_chars))

        # 优化句子：替换重复字，增强意象组合
        optimized_lines = []
        for line in lines:
            optimized_line = self._optimize_line(line, theme, keyword)
            optimized_lines.append(optimized_line)

        # 添加标点并调整句式
        formatted_lines = []
        for i, line in enumerate(optimized_lines):
            if line:
                # 避免句子以标点开头
                if line[0] in self.punctuation:
                    line = random.choice(self.fallback_chars) + line[1:]
                # 添加标点（偶数句用逗号，奇数句用句号）
                punc = "，" if i < len(optimized_lines) - 1 and i % 2 == 0 else "。"
                formatted_lines.append(line + punc)

        # 生成标题
        title = f"《{rhythm_name}·{keyword}》\n\n"
        poem_body = "\n".join(formatted_lines)

        return title + poem_body

    def _optimize_line(self, line, theme, keyword):
        """优化单个句子，严格控制重复字，增强意象"""
        # 替换重复字（严格版）
        char_count = Counter(line)
        for char, count in char_count.items():
            # 关键词允许出现2次，其他字最多出现2次
            if count > (2 if char == keyword else 2) and char in self.common_chars:
                replacement = random.choice([c for c in self.fallback_chars if c != char and c != keyword])
                line = line.replace(char, replacement, count - 2)  # 保留2次

        # 增强主题相关意象（优先使用意象组合，避免重复）
        if theme in self.image_combinations:
            used_images = set()
            for img1, img2 in self.image_combinations[theme]:
                # 检查是否已使用该意象且句子长度允许
                if img1[0] not in used_images and img2[0] not in used_images and len(line) < 12:
                    if img1[0] in line and img2[0] not in line:
                        line = line.replace(img1[0], img1, 1)
                        line = line + img2 if len(line) < 12 else line
                        used_images.add(img1[0])
                        used_images.add(img2[0])
                    elif img1[0] not in line and img2[0] in line:
                        line = img1 + line.replace(img2[0], img2, 1) if len(line) < 12 else line
                        used_images.add(img1[0])
                        used_images.add(img2[0])

        return line

# 主函数
def main():
    torch.cuda.empty_cache()
    gc.collect()

    data_dir = 'data'
    min_freq = 2

    print("开始数据预处理...")
    preprocessor = DataPreprocess(data_dir, min_freq)

    # 加载全部数据
    train_songs, val_songs, all_keywords, all_authors, all_rhythms = preprocessor.load_data(sample_data=None)

    if len(train_songs) < 2 or len(val_songs) < 1:
        print("错误: 有效数据不足，无法继续训练")
        return

    # 构建词表
    all_texts = train_songs + val_songs
    vocab = preprocessor.build_vocab(all_texts)

    # 处理序列
    train_sequences = preprocessor.process_sequences(train_songs)
    val_sequences = preprocessor.process_sequences(val_songs)

    print(f"词汇表大小: {len(vocab)}")
    print(f"最大序列长度: {preprocessor.max_seq_length}")

    word2idx = preprocessor.word2idx
    idx2word = preprocessor.idx2word

    # 创建数据加载器
    train_dataset = TensorDataset(torch.tensor(train_sequences, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(val_sequences, dtype=torch.long))
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 256

    # 初始化模型
    generator = Seq2SeqGenerator(vocab_size, embedding_dim, hidden_dim, word2idx=word2idx).to(device)
    discriminator = LightweightDiscriminator(vocab_size, embedding_dim, hidden_dim).to(device)

    total_g_params = sum(p.numel() for p in generator.parameters())
    total_d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"生成器参数总数: {total_g_params:,}")
    print(f"判别器参数总数: {total_d_params:,}")

    # 初始化训练器
    trainer = EnhancedGANTrainer(
        generator, discriminator, word2idx, idx2word, device, preprocessor.rhythm_patterns
    )

    print("开始训练模型...")
    try:
        best_val_loss = trainer.train_gan(train_dataloader, val_dataloader, epochs=100, d_steps=3)
        print("模型训练完成!")
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("\n内存不足! 减小批大小...")
            train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
            best_val_loss = trainer.train_gan(train_dataloader, val_dataloader, epochs=100, d_steps=3)
            print("模型训练完成!")
        else:
            raise e

    if os.path.exists('best_gan_model.pth'):
        try:
            # 加载完整的检查点
            checkpoint = torch.load('best_gan_model.pth', map_location=device,weights_only=False)
            # 只加载生成器的权重
            generator.load_state_dict(checkpoint['generator'])
            print(f"加载最佳模型 (损失: {checkpoint['loss']:.4f})")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("尝试使用默认初始化模型...")

    print("\n宋词生成器已就绪，请输入关键词生成宋词（输入'exit'退出）")
    while True:
        keyword = input("\n请输入关键词(如'春','秋','月','乡思'等): ")

        if keyword.lower() == 'exit':
            print("感谢使用宋词生成器，再见！")
            break

        print(f"\n生成宋词中(关键词: '{keyword}')...")
        start_time = time.time()
        poem = trainer.generate_poem(keyword, temperature=0.7)
        gen_time = time.time() - start_time

        print(f"\n{poem}")
        print(f"\n生成耗时: {gen_time:.2f}秒")
        print('=' * 60)

if __name__ == "__main__":
    main()
    torch.cuda.empty_cache()
    gc.collect()