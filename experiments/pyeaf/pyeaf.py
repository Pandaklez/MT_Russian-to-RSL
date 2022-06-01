import pympi
import os
import re
import copy


class EAFReader(object):
	# возможность того, что есть второй человечек НЕ учтена
	
    RUS = "Перевод"
    RSL_R = "ПР-глосс"
    RSL_L = "ЛР-глосс"
    NEW = "Новое"
    DT = 5  # погрешность 2

    @staticmethod
    def get_files_from_folder(path):
        path = path.rstrip('/')
        files = list(map(lambda x: f'{path}/' + x, filter(lambda x: x.endswith('.eaf'), os.listdir(f'{path}/'))))
        return files

    def __init__(self, directory=None, file=None):
        self.dir = directory
        self.file = file
        if self.dir:
            self.files = self.get_files_from_folder(self.dir)
        elif self.file:
            self.files = [self.file]
        else:
            raise ValueError("Argument dir and file is not correct")
        self._data = None
        
    def _sentence_boarders(self, tmp_tier):
        start_ind, end_ind = False, False
                    
        starting = [k for k, annotation in enumerate(tmp_tier) if annotation[2].startswith('№')]
        ending = [k for k, annotation in enumerate(tmp_tier) if annotation[2].endswith('%')]
        
        if starting:
            start_ind = starting[0]   # это общий случай нормальный
            # Если разметчик допустил ошибку и не отметил окончание, но отметил начало следующего предложения
            if not ending:
                if len(starting) > 1:
                    end_ind = starting[1] - 1  # -1 потому что само начальное слово не хотим брать
                else:
                    # А что если погрешности не хватило, чтобы собрать все глоссы?
                    end_ind = -1   
            else:   # это общий случай нормальный
                for el_ind in ending:
                    if el_ind >= start_ind:  # end_ind > start_ind, но самый крайний слева при этом 
                        end_ind = el_ind
                        break
        else:   # Если разметчик допустил ошибку и не разметил начало
            if ending:
                start_ind = 0
                for el_ind in ending:
                    if el_ind >= start_ind:
                        end_ind = el_ind
                        break
                if len(ending) > 1:
                    start_ind = ending[0] + 1  # +1 потому что хотим отрезать концовку предыдущего предложения
            else:   # Если нет ни начала ни конца предложения в пределах погрешности
                start_ind = 0
                end_ind = -1
        
        return start_ind, end_ind
                    

    def _file_preparation(self, fname):
        eaf = pympi.Eaf(fname)
        tiers = eaf.get_tier_names()
        pairs = []

        if self.RUS in tiers:
            russian = eaf.get_annotation_data_for_tier(self.RUS)
            for t_begin, t_end, phrase in russian:
                sentence = {self.RUS: phrase}

                if self.RSL_R in tiers:
                    tmp_r = eaf.get_annotation_data_between_times(self.RSL_R, t_begin - self.DT, t_end + self.DT)
                    if not tmp_r:
                        sentence.update({self.RSL_R: []})
                    else:
                        start_ind, end_ind = self._sentence_boarders(tmp_r)
                        tmp_r_clean = list(map(lambda x: x[2].strip('№%'), tmp_r))

                        # нашлось ли вообще предложение в этом таймстемпе
                        if isinstance(start_ind, int) and isinstance(end_ind, int):
                            if start_ind == end_ind:  # в предложении ровно одна глосса
                                sentence.update({self.RSL_R: [tmp_r_clean[start_ind]]})  # вторые скобочки нужны для правильной токенизации потом
                            else:
                                sentence.update({self.RSL_R: tmp_r_clean[start_ind:end_ind+1]})
                        else:  # таких случаев не должно быть вообще
                            print(list(map(lambda x: x[2], tmp_r)))
                            print('ERROR end_ind start_ind not a number !!!!\n')

                if self.RSL_L in tiers:
                    tmp_l = eaf.get_annotation_data_between_times(self.RSL_L, t_begin - self.DT, t_end + self.DT)
                    if not tmp_l:
                        sentence.update({self.RSL_L: []})
                        
                    sentence.update({self.RSL_L: list(map(lambda x: x[2], tmp_l))})
                pairs.append(sentence)
        #print(pairs[:10])
        return pairs

    def load(self):
        sentences = []
        for f in self.files:
            sentences += self._file_preparation(f)

        self._data = copy.copy(sentences)
        return self

    @property
    def data(self):
        if not self._data:
            self.load()
        return self._data

    def get_sentences(self, tier):
        return [d[tier] for d in self.data if tier in d]
