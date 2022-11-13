from SoccerNet.Downloader import SoccerNetDownloader
from SoccerNet.utils import getListGames


def download_games():
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="path/to/soccernet")

    # Download SoccerNet labels
    mySoccerNetDownloader.downloadGames(files=["Labels.json"], split=["train","valid","test"]) # download labels
    mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"]) # download labels SN v2
    mySoccerNetDownloader.downloadGames(files=["Labels-cameras.json"], split=["train","valid","test"]) # download labels for camera shot

    # Download SoccerNet features
    mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["train","valid","test"]) # download Features
    mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["train","valid","test"]) # download Features reduced with PCA
    mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["train","valid","test"]) # download Player Bounding Boxes inferred with MaskRCNN
    mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["train","valid","test"]) # download Field Calibration inferred with CCBV
    mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy","2_baidu_soccer_embeddings.npy"], split=["train","valid","test"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports

    # Download SoccerNet videos (require password from NDA to download videos)
    mySoccerNetDownloader.password = input("Password for videos? (contact the author):\n")
    mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train","valid","test"]) # download 224p Videos
    mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test"]) # download 720p Videos

    # Download SoccerNet Challenge set (require password from NDA to download videos)
    mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2.npy", "2_ResNET_TF2.npy"], split=["challenge"]) # download ResNET Features
    mySoccerNetDownloader.downloadGames(files=["1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy"], split=["challenge"]) # download ResNET Features reduced with PCA
    mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["challenge"]) # download 224p Videos (require password from NDA)
    mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["challenge"]) # download 720p Videos (require password from NDA)
    mySoccerNetDownloader.downloadGames(files=["1_player_boundingbox_maskrcnn.json", "2_player_boundingbox_maskrcnn.json"], split=["challenge"]) # download Player Bounding Boxes inferred with MaskRCNN
    mySoccerNetDownloader.downloadGames(files=["1_field_calib_ccbv.json", "2_field_calib_ccbv.json"], split=["challenge"]) # download Field Calibration inferred with CCBV
    mySoccerNetDownloader.downloadGames(files=["1_baidu_soccer_embeddings.npy","2_baidu_soccer_embeddings.npy"], split=["challenge"]) # download Frame Embeddings from https://github.com/baidu-research/vidpress-sports

    return mySoccerNetDownloader


def list_games():
    print(getListGames(split="train"))  # return list of games recommended for training
    print(getListGames(split="valid"))  # return list of games recommended for validation
    print(getListGames(split="test"))  # return list of games recommended for testing
    print(getListGames(split="challenge"))  # return list of games recommended for challenge
    print(getListGames(
        split=["train", "valid", "test", "challenge"]))  # return list of games for training, validation and testing
    print(getListGames(split="v1"))  # return list of games from SoccerNetv1 (train/valid/test)


def download_tracking_data():
    mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="./data")
    mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])


if __name__ == '__main__':
    download_tracking_data()
