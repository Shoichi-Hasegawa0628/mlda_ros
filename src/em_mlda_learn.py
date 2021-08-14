#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __init__ import *
import pylab
import pickle
import sys
import codecs


class MLDA():
    def load_model(self, estimate_mode):
        with open(DATA_FOLDER + "/learn_result/model.pickle", "rb") as f:
            a, b, c, d, e = pickle.load(f)

        if estimate_mode:
            return b, c
        else:
            return a, b, c, d, e

    def save_model(self, save_dir, n_dz, n_mzw, n_mz, docs_mdn, topics_mdn, M, dims):
        rospy.loginfo("[Service em_mlda/learn] Savinng the laerning result.")

        try:
            os.mkdir(save_dir)
        except:
            pass

        Pdz = n_dz + ALPHA
        Pdz = (Pdz.T / Pdz.sum(1)).T
        np.savetxt(os.path.join(save_dir, "Pdz.txt"),
                   Pdz, delimiter=",", fmt=str("%f"))

        for m in range(M):
            Pwz = (n_mzw[m].T + BETA) / (n_mz[m] + dims[m] * BETA)
            Pdw = Pdz.dot(Pwz.T)
            np.savetxt(os.path.join(
                save_dir, "Pmdw[%d].txt" % m), Pdw, fmt=str("%f"))

        with open(os.path.join(save_dir, "model.pickle"), "wb") as f:
            pickle.dump([n_dz, n_mzw, n_mz, docs_mdn, topics_mdn], f)

        rospy.loginfo("[Service em_mlda/learn] Finish the saving the model.")

    # 尤度計算
    def calc_liklihood(self, target_modality_num, n_dz, n_zw, n_z, K, V):
        lik = 0
        P_wz = (n_zw.T + BETA) / (n_z + V * BETA)

        if self.data[target_modality_num].ndim == 1:
            object_num = 1
        else:
            object_num = len(self.data[target_modality_num])

        for d in range(object_num):
            Pz = (n_dz[d] + ALPHA) / (np.sum(n_dz[d]) + K * ALPHA)
            Pwz = Pz * P_wz
            Pw = np.sum(Pwz, 1) + 0.000001
            lik += np.sum(self.data[target_modality_num][d] * np.log(Pw))

        return lik

    # 単語を一列に並べたリスト変換
    def conv_to_word_list(self, data):
        if data.ndim == 1:
            V = data.shape[0]
        else:
            V = len(data)

        doc = []
        for v in range(V):  # v:語彙のインデックス
            # 語彙の発生した回数文forを回す
            for n in range(data[v]):
                doc.append(v)

        return doc

    def sample_topic(self, target_object, Target_character_index, n_dz, n_zw, n_z, dimension_list):
        P = [0.0] * CATEGORYNUM

        # 累積確率を計算
        P = (n_dz[target_object, :] + ALPHA) * (n_zw[:,
                                                     Target_character_index] + BETA) / (n_z[:] + dimension_list * BETA)
        for z in range(1, CATEGORYNUM):
            P[z] = P[z] + P[z - 1]

        # サンプリング
        rnd = P[CATEGORYNUM - 1] * random.random()
        for z in range(CATEGORYNUM):
            if P[z] >= rnd:
                return z

    def calc_lda_param(self, docs_mdn, topics_mdn, dims):
        Modality_num = len(docs_mdn)
        Object_num = len(docs_mdn[0])

        # 各物体dにおいてカテゴリzが発生した回数
        n_dz = np.zeros((Object_num, CATEGORYNUM))

        # 各カテゴリzにおいてモダリティーごとに特徴wが発生した回数
        n_mzw = [np.zeros((CATEGORYNUM, dims[m])) for m in range(Modality_num)]

        # モダリティーごとに各トピックが発生した回数
        n_mz = [np.zeros(CATEGORYNUM) for m in range(Modality_num)]

        # 数え上げる
        for d in range(Object_num):
            for m in range(Modality_num):
                if dims[m] == 0:
                    continue
                N = len(docs_mdn[m][d])  # 物体に含まれる特徴数
                for n in range(N):
                    w = docs_mdn[m][d][n]  # 物体dのn番目の特徴のインデックス
                    z = topics_mdn[m][d][n]  # 特徴に割り当てれれているトピック
                    n_dz[d][z] += 1
                    n_mzw[m][z][w] += 1
                    n_mz[m][z] += 1

        return n_dz, n_mzw, n_mz

    def plot(self, n_dz, liks, D):
        print("対数尤度：", liks[-1])
        doc_dopics = np.argmax(n_dz, 1)
        print("分類結果：", doc_dopics)
        print("---------------------")

        # グラフ表示
        pylab.clf()
        pylab.subplot("121")
        pylab.title("P(z|d)")
        pylab.imshow(n_dz / np.tile(np.sum(n_dz, 1).reshape(D, 1),
                                    (1, CATEGORYNUM)), interpolation="none")
        pylab.subplot("122")
        pylab.title("liklihood")
        pylab.plot(list(range(len(liks))), liks)
        pylab.draw()
        pylab.pause(0.01)

    def mlda_learn(self, save_dir="model", estimate_mode=False):
        rospy.loginfo("[Service em_mlda/learn] start to learning")

        pylab.ion()  # インタラクティブモード(コンソール入力がいつでもできる)

        liks = []  # 尤度のリスト
        Modality_num = len(self.data)  # モダリティ数(サンプルは2つ)

        dimension_list = []  # 各モダリティごとの次元数を保存する

        for Target_modality in range(Modality_num):
            if self.data[Target_modality] is not None:
                if self.data[Target_modality].ndim == 1:
                    dimension_list.append(self.data[Target_modality].shape[0])
                    Object_num = 1
                else:
                    # 次元数をdimension_listに追加
                    dimension_list.append(len(self.data[Target_modality][0]))
                    Object_num = len(self.data[Target_modality])  # 物体数
            else:
                dimension_list.append(0)

        # [NoneがObject_num個]がModality_num個
        docs_mdn = [[None for i in range(Object_num)]
                    for m in range(Modality_num)]
        topics_mdn = [[None for i in range(Object_num)]
                      for m in range(Modality_num)]

        # data内の単語を一列に並べる（計算しやすくするため）
        for Target_object in range(Object_num):
            for Target_modality in range(Modality_num):
                if self.data[Target_modality] is not None:
                    if Object_num == 1:
                        docs_mdn[Target_modality][Target_object] = self.conv_to_word_list(
                            self.data[Target_modality])
                        topics_mdn[Target_modality][Target_object] = np.random.randint(
                            0, CATEGORYNUM, len(docs_mdn[Target_modality][Target_object]))  # 各単語にランダムでトピックを割り当てる

                    else:
                        docs_mdn[Target_modality][Target_object] = self.conv_to_word_list(
                            self.data[Target_modality][Target_object])
                        topics_mdn[Target_modality][Target_object] = np.random.randint(
                            0, CATEGORYNUM, len(docs_mdn[Target_modality][Target_object]))  # 各単語にランダムでトピックを割り当てる

        # LDAのパラメータを計算
        n_dz, n_mzw, n_mz = self.calc_lda_param(
            docs_mdn, topics_mdn, dimension_list)

        if estimate_mode:
            n_mzw, n_mz = self.load_model(True)

        for It in range(ITERATION):
            # メインの処理
            for Target_object in range(Object_num):
                for Target_modality in range(Modality_num):
                    if self.data[Target_modality] is None:
                        continue

                    Target_character_num = len(
                        docs_mdn[Target_modality][Target_object])  # 物体dのモダリティmに含まれる特徴数

                    for Target_character in range(Target_character_num):
                        # 特徴のインデックス
                        Target_character_index = docs_mdn[Target_modality][Target_object][Target_character]
                        # 特徴に割り当てられているカテゴリ
                        Target_character_category = topics_mdn[Target_modality][Target_object][Target_character]
                        # データを取り除きパラメータを更新
                        n_dz[Target_object][Target_character_category] -= 1

                        if not estimate_mode:
                            n_mzw[Target_modality][Target_character_category][Target_character_index] -= 1
                            n_mz[Target_modality][Target_character_category] -= 1

                            # サンプリング
                        Target_character_category = self.sample_topic(
                            Target_object, Target_character_index, n_dz, n_mzw[Target_modality], n_mz[Target_modality], dimension_list[Target_modality])

                        # データをサンプリングされたクラスに追加してパラメータを更新
                        topics_mdn[Target_modality][Target_object][Target_character] = Target_character_category
                        n_dz[Target_object][Target_character_category] += 1

                        if not estimate_mode:
                            n_mzw[Target_modality][Target_character_category][Target_character_index] += 1
                            n_mz[Target_modality][Target_character_category] += 1

            lik = 0

            for Target_modality in range(Modality_num):
                if self.data[Target_modality] is not None:
                    lik += self.calc_liklihood(Target_modality, n_dz, n_mzw[Target_modality],
                                               n_mz[Target_modality], CATEGORYNUM, dimension_list[Target_modality])
            liks.append(lik)
            # self.plot(n_dz, liks, Object_num)

            if It == ITERATION - 1:
            # if True:
                print("Iteration ", It + 1)

                print u'対数尤度：'
                print(liks[-1])

                doc_dopics = np.argmax(n_dz, 1)
                print u'分類結果：'
                print(doc_dopics)

                print("---------------------")

        self.save_model(save_dir, n_dz, n_mzw, n_mz, docs_mdn,
                        topics_mdn, Modality_num, dimension_list)

    def mlda_server(self, req):
        if req.status == "learn":
            self.data = [np.loadtxt(DATA_FOLDER + "/histogram_image.txt", dtype=np.int32),
                         np.loadtxt(DATA_FOLDER +
                                    "/histogram_hardness.txt", dtype=np.int32),
                         np.loadtxt(DATA_FOLDER +
                                    "/histogram_weight.txt", dtype=np.int32),
                         np.loadtxt(DATA_FOLDER + "/histogram_word.txt", dtype=np.int32) * 10]

            if ESTIMATE:
                rospy.loginfo("[Service em_mlda/learn] Re-learning mode start")
                self.mlda_learn(DATA_FOLDER + "/learn_result", False)
            else:
                rospy.loginfo(
                    "[Service em_mlda/learn] Defalut learning mode start")
                self.mlda_learn(DATA_FOLDER + "/learn_result", False)

        elif req.status == "estimate":
            self.data = [np.loadtxt(DATA_FOLDER + "/unknown_histogram_image" + str(req.count) + ".txt", dtype=np.int32),
                         np.loadtxt(DATA_FOLDER + "/unknown_histogram_hardness" +
                                    str(req.count) + ".txt", dtype=np.int32),
                         np.loadtxt(DATA_FOLDER + "/unknown_histogram_weight" +
                                    str(req.count) + ".txt", dtype=np.int32),
                         None]

            rospy.loginfo(
                "[Service em_mlda/learn] Estimate unknown object mode start")
            self.mlda_learn(DATA_FOLDER + "/estimate_result" +
                            str(req.count), True)

        else:
            rospy.logwarn(
                "[Service em_mlda/learn] command not found: %s", req.status)

        return mlda_learnResponse(True)

    def __init__(self):
        s = rospy.Service('em_mlda/learn', mlda_learn, self.mlda_server)
        rospy.loginfo("[Service em_mlda/learn] Ready em_mlda/learn")

        self.data = []


if __name__ == '__main__':
    sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
    rospy.init_node('em_mlda_learn_server')
    MLDA()
    rospy.spin()
