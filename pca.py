from sklearn.decomposition import PCA


def pca(param_array):

    pca = PCA(n_components=1)
    pca_array = pca.fit_transform(param_array)

    # 시각화
    #plt.figure(figsize=(8, 6))
    #plt.scatter(pca_array, [0]*len(pca_array), alpha=0.5)
    #plt.title('PCA')
    #plt.xlabel('PCA')
    #plt.yticks([])  # Y축 값 숨기기
    #plt.show()

    return pca_array;