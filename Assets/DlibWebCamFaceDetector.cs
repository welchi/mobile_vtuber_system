using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using DlibFaceLandmarkDetector;
using DlibFaceLandmarkDetector.UnityUtils;

public class DlibWebCamFaceDetector : MonoBehaviour
{
    // トラッキング状況を表示するRawImage
    public GameObject surface;

    // トラッキング関連
    [SerializeField] int requestedWidth = 320;
    [SerializeField] int requestedHeight = 240;
    [SerializeField] int requestedFps = 30;
    [SerializeField] bool requestedIsFrontFacing = true;

    // WebCamTexture関連
    private WebCamTexture _webCamTexture;
    private WebCamDevice _webCamDevice;
    private Color32[] _colors;
    private Color32[] _rotatedColors;
    private bool _rotate90Degree = false;
    private bool _isInitWaiting = false;
    private bool _hasInitDone = false;
    private ScreenOrientation _screenOrientation;
    private int _screenWidth;
    private int _screenHeight;

    // Landmark関連
    private FaceLandmarkDetector faceLandmarkDetector;
    private Texture2D _texture;
    private string _dlibShapePredictorFileName = "sp_human_face_68.dat";
    private string _dlibShapePredictorFilePath;
    private List<Vector2> _landmarks;

    public List<Vector2> Landmarks
    {
        get { return _landmarks; }
        set { _landmarks = value; }
    }

    void Start()
    {
        _dlibShapePredictorFilePath = Utils.getFilePath(_dlibShapePredictorFileName);
        Run();
    }

    private void Run()
    {
        if (string.IsNullOrEmpty(_dlibShapePredictorFilePath))
            Debug.LogError(
                "shape predictor ファイルが見つかりません。 “DlibFaceLandmarkDetector/StreamingAssets/” を “Assets/StreamingAssets/” へ置いてください。");

        faceLandmarkDetector = new FaceLandmarkDetector(_dlibShapePredictorFilePath);
        Initialize();
    }

    /// <summary>
    /// 初期化処理
    /// </summary>
    private void Initialize()
    {
        if (_isInitWaiting)
            return;
        StartCoroutine(_Initialize());
    }

    private IEnumerator _Initialize()
    {
        if (_hasInitDone)
            Dispose();

        _isInitWaiting = true;

        // フロントカメラを取得
        if (_webCamTexture == null)
        {
            for (int cameraIndex = 0; cameraIndex < WebCamTexture.devices.Length; cameraIndex++)
            {
                if (WebCamTexture.devices[cameraIndex].isFrontFacing == requestedIsFrontFacing)
                {
                    _webCamDevice = WebCamTexture.devices[cameraIndex];
                    _webCamTexture =
                        new WebCamTexture(_webCamDevice.name, requestedWidth, requestedHeight, requestedFps);
                    break;
                }
            }
        }

        if (_webCamTexture == null)
        {
            Debug.LogError("デバイスにフロントカメラが存在しませんでした。");
            _isInitWaiting = false;
            yield break;
        }

        // カメラ再生
        _webCamTexture.Play();
        while (true)
        {
            // 現フレームのバッファが更新されるまで待つ
            if (_webCamTexture.didUpdateThisFrame)
            {
                _screenOrientation = Screen.orientation;
                _screenWidth = Screen.width;
                _screenHeight = Screen.height;
                _isInitWaiting = false;
                _hasInitDone = true;

                OnInited();
                break;
            }
            else
            {
                yield return 0;
            }
        }
    }

    /// <summary>
    /// WebCamTexture初期化
    /// </summary>
    private void OnInited()
    {
        // 色を初期化
        if (_colors == null || _colors.Length != _webCamTexture.width * _webCamTexture.height)
        {
            _colors = new Color32[_webCamTexture.width * _webCamTexture.height];
            _rotatedColors = new Color32[_webCamTexture.width * _webCamTexture.height];
        }

        // トラッキング画像を画面の向きに合わせる
        if (Screen.orientation == ScreenOrientation.Portrait ||
            Screen.orientation == ScreenOrientation.PortraitUpsideDown)
            _rotate90Degree = true;
        else
            _rotate90Degree = false;
        if (_rotate90Degree)
            _texture = new Texture2D(_webCamTexture.height, _webCamTexture.width, TextureFormat.RGBA32, false);
        else
            _texture = new Texture2D(_webCamTexture.width, _webCamTexture.height, TextureFormat.RGBA32, false);

        surface.GetComponent<RawImage>().texture = _texture;
        surface.GetComponent<RectTransform>().sizeDelta = new Vector2(320, 240);
    }

    void Update()
    {
        // 画面の回転を取得
        // 傾きが変わった場合
        if (_screenOrientation != Screen.orientation &&
            (_screenWidth != Screen.width || _screenHeight != Screen.height))
        {
            Initialize();
        }
        else
        {
            _screenWidth = Screen.width;
            _screenHeight = Screen.height;
        }

        // ランドマーク推定
        if (_hasInitDone && _webCamTexture.isPlaying && _webCamTexture.didUpdateThisFrame)
        {
            Color32[] colors = GetColors();
            if (colors != null)
            {
                faceLandmarkDetector.SetImage<Color32>(colors, _texture.width, _texture.height, 4, true);

                // 顔が含まれる領域を取得
                List<Rect> detectResult = faceLandmarkDetector.Detect();

                foreach (var rect in detectResult)
                {
                    // 顔認識(ランドマーク推定)
                    Landmarks = faceLandmarkDetector.DetectLandmark(rect);
                    // ランドマークを描画
                    faceLandmarkDetector.DrawDetectLandmarkResult<Color32>(colors, _texture.width, _texture.height, 4,
                        true, 0, 255, 0, 255);
                }

                // 顔領域を矩形で描画
                faceLandmarkDetector.DrawDetectResult<Color32>(colors, _texture.width, _texture.height, 4, true, 255, 0,
                    0, 255, 2);

                _texture.SetPixels32(colors);
                _texture.Apply(false);
            }
        }
    }

    /// <summary>
    /// WebCamTextureが映した画像をRGBAとして取得
    /// </summary>
    private Color32[] GetColors()
    {
        _webCamTexture.GetPixels32(_colors);


        if (_rotate90Degree)
        {
            Rotate90CW(_colors, _rotatedColors, _webCamTexture.width, _webCamTexture.height);
            FlipColors(_rotatedColors, _webCamTexture.width, _webCamTexture.height);
            return _rotatedColors;
        }
        else
        {
            FlipColors(_colors, _webCamTexture.width, _webCamTexture.height);
            return _colors;
        }

        return _colors;
    }


    /// <summary>
    /// 映像を反転
    /// </summary>
    void FlipColors(Color32[] colors, int width, int height)
    {
        int flipCode = int.MinValue;

        if (_webCamDevice.isFrontFacing)
        {
            if (_webCamTexture.videoRotationAngle == 0)
                flipCode = 1;
            else if (_webCamTexture.videoRotationAngle == 90)
                flipCode = 1;
            else if (_webCamTexture.videoRotationAngle == 180)
                flipCode = 0;
            else if (_webCamTexture.videoRotationAngle == 270)
                flipCode = 0;
        }

        // 反転が必要な場合
        if (flipCode > int.MinValue)
        {
            if (_rotate90Degree)
            {
                if (flipCode == 0)
                    FlipVertical(colors, colors, height, width);
                else if (flipCode == 1)
                    FlipHorizontal(colors, colors, height, width);
                else if (flipCode < 0)
                    Rotate180(colors, colors, height, width);
            }
            else
            {
                if (flipCode == 0)
                    FlipVertical(colors, colors, width, height);
                else if (flipCode == 1)
                    FlipHorizontal(colors, colors, width, height);
                else if (flipCode < 0)
                    Rotate180(colors, colors, height, width);
            }
        }
    }

    /// <summary>
    /// 映像を反転(縦)
    /// </summary>
    void FlipVertical(Color32[] src, Color32[] dst, int width, int height)
    {
        for (var i = 0; i < height / 2; i++)
        {
            var y = i * width;
            var x = (height - i - 1) * width;
            for (var j = 0; j < width; j++)
            {
                int s = y + j;
                int t = x + j;
                Color32 c = src[s];
                dst[s] = src[t];
                dst[t] = c;
            }
        }
    }

    /// <summary>
    /// 映像を反転(横)
    /// </summary>
    void FlipHorizontal(Color32[] src, Color32[] dst, int width, int height)
    {
        for (int i = 0; i < height; i++)
        {
            int y = i * width;
            int x = y + width - 1;
            for (var j = 0; j < width / 2; j++)
            {
                int s = y + j;
                int t = x - j;
                Color32 c = src[s];
                dst[s] = src[t];
                dst[t] = c;
            }
        }
    }

    /// <summary>
    /// 映像を180度回転
    /// </summary>
    void Rotate180(Color32[] src, Color32[] dst, int height, int width)
    {
        int i = src.Length;
        for (int x = 0; x < i / 2; x++)
        {
            Color32 t = src[x];
            dst[x] = src[i - x - 1];
            dst[i - x - 1] = t;
        }
    }

    /// <summary>
    /// 映像を時計回りに90度回転
    /// </summary>
    void Rotate90CW(Color32[] src, Color32[] dst, int height, int width)
    {
        int i = 0;
        for (int x = height - 1; x >= 0; x--)
        {
            for (int y = 0; y < width; y++)
            {
                dst[i] = src[x + y * height];
                i++;
            }
        }
    }

    /// <summary>
    /// Destroyイベントの処理
    /// </summary>
    private void OnDestroy()
    {
        Dispose();
        if (faceLandmarkDetector != null)
            faceLandmarkDetector.Dispose();
    }

    /// <summary>
    /// すべてのリソースを解放
    /// </summary>
    private void Dispose()
    {
        _rotate90Degree = false;
        _isInitWaiting = false;
        _hasInitDone = false;
        if (_webCamTexture != null)
        {
            _webCamTexture.Stop();
            Destroy(_webCamTexture);
            _webCamTexture = null;
        }

        if (_texture != null)
        {
            Destroy(_texture);
            _texture = null;
        }
    }
}