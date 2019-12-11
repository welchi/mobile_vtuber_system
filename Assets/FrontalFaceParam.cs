using UnityEngine;
using System;
using System.Collections.Generic;
using Live2D.Cubism.Core;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.UnityUtils;
using static DlibWebCamFaceDetector;

public class FrontalFaceParam : MonoBehaviour
{
    [SerializeField] DlibWebCamFaceDetector faceDetector;
    [SerializeField] CubismParameter headAngleParameterX;
    [SerializeField] CubismParameter headAngleParameterY;
    [SerializeField] CubismParameter headAngleParameterZ;
    [SerializeField] float lerpT = 0.2f;
    private Vector3 _headRotation;

    // 座標変換関連
    private MatOfPoint3f _objectPoints;
    private MatOfPoint2f _imagePoints;
    private Mat _rotM;
    private Mat _camMatrix;
    private MatOfDouble _distCoeffs;
    private Matrix4x4 _invertYM;
    private Matrix4x4 _invertZM;
    private Matrix4x4 _transformationM;
    private Mat _rVec;
    private Mat _tVec;
    private Matrix4x4 _ARM;
    private PoseData _oldPoseData;
    private Matrix4x4 _VP;

    // 正規化関連
    [SerializeField] float normWidth = 200;
    [SerializeField] float normHeight = 200;
    List<Vector2> _normPoints;

    // ローパスフィルタ関連
    [SerializeField] float positionLowPass = 4f;
    [SerializeField] float rotationLowPass = 2f;

    void Awake()
    {
        _invertYM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, -1, 1));
        _invertZM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, 1, -1));

        // 顔の初期位置を設定
        _objectPoints = new MatOfPoint3f(
            new Point3(-31, 72, 86), // 左目
            new Point3(31, 72, 86), // 右目
            new Point3(0, 40, 114), // 鼻
            new Point3(-20, 15, 90), // 左口角
            new Point3(20, 15, 90), // 右口角
            new Point3(-69, 76, -2), // 左耳
            new Point3(69, 76, -2) // 右耳
        );
        _imagePoints = new MatOfPoint2f();
        _rotM = new Mat(3, 3, CvType.CV_64FC1);

        // カメラの内部パラメータ
        float maxD = Mathf.Max(normHeight, normWidth);
        float fx = maxD;
        float fy = maxD;
        float cx = normWidth / 2.0f;
        float cy = normHeight / 2.0f;
        _camMatrix = new Mat(3, 3, CvType.CV_64FC1);
        _camMatrix.put(0, 0, fx);
        _camMatrix.put(0, 1, 0);
        _camMatrix.put(0, 2, cx);
        _camMatrix.put(1, 0, 0);
        _camMatrix.put(1, 1, fy);
        _camMatrix.put(1, 2, cy);
        _camMatrix.put(2, 0, 0);
        _camMatrix.put(2, 1, 0);
        _camMatrix.put(2, 2, 1.0f);

        _distCoeffs = new MatOfDouble(0, 0, 0, 0);


        // カメラキャリブレーション
        Matrix4x4 P = ARUtils.CalculateProjectionMatrixFromCameraMatrixValues((float) fx, (float) fy, (float) cx,
            (float) cy, normWidth, normHeight, 0.3f, 2000f);
        Matrix4x4 V = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(1, 1, -1));
        _VP = P * V;

        _normPoints = new List<Vector2>(68);
        for (int i = 0; i < 68; i++)
            _normPoints.Add(new Vector2(0, 0));
    }

    private void LateUpdate()
    {
        var landmarks = faceDetector.Landmarks;
        Vector3 angles = GetFrontalFaceAngle(landmarks);
        // 検出ミスしたら補正をかける
        float angleX = (angles.x > 180) ? angles.x - 360 : angles.x;
        float angleY = (angles.y > 180) ? angles.y - 360 : angles.y;
        float angleZ = (angles.z > 180) ? angles.z - 360 : angles.z;
        angleX = (angles.x < -180) ? angles.x + 360 : angles.x;
        angleY = (angles.y < -180) ? angles.y + 360 : angles.y;
        angleZ = (angles.z < -180) ? angles.z + 360 : angles.z;
        // 座標系を変換
        float temp = angleX;
        angleX = -angleY;
        angleY = temp;
        angleZ = -angleZ;
        
        // 線形補間によるスムージング
        _headRotation=new Vector3(
            Mathf.LerpAngle(_headRotation.x,angleX,lerpT),
            Mathf.LerpAngle(_headRotation.y,angleY,lerpT),
            Mathf.LerpAngle(_headRotation.z,angleZ,lerpT));
        
        SetParameter(headAngleParameterX,_headRotation.x);
        SetParameter(headAngleParameterY,_headRotation.y);
        SetParameter(headAngleParameterZ,_headRotation.z);
        
    }

    /// <summary>
    /// 顔の向きを取得
    /// </summary>
    private Vector3 GetFrontalFaceAngle(List<Vector2> points)
    {
        if (points.Count != 68)
            throw new ArgumentNullException("ランドマークが正しくありません。");

        if (_camMatrix == null)
            throw new ArgumentNullException("カメラの内部パラメータが正しくありません。");

        // スケールの正規化
        float normScale = Math.Abs(points[30].y - points[8].y) / (normHeight / 2);
        Vector2 normDiff = points[30] * normScale - new Vector2(normWidth / 2, normHeight / 2);

        for (int i = 0; i < points.Count; i++)
            _normPoints[i] = points[i] * normScale - normDiff;

        _imagePoints.fromArray(
            new Point((_normPoints[38].x + _normPoints[41].x) / 2, (_normPoints[38].y + _normPoints[41].y) / 2), // 左目
            new Point((_normPoints[43].x + _normPoints[46].x) / 2, (_normPoints[43].y + _normPoints[46].y) / 2), // 右目
            new Point(_normPoints[33].x, _normPoints[33].y), // 鼻
            new Point(_normPoints[48].x, _normPoints[48].y), // 左口角
            new Point(_normPoints[54].x, _normPoints[54].y), // 右口角
            new Point(_normPoints[0].x, _normPoints[0].y), // 左耳
            new Point(_normPoints[16].x, _normPoints[16].y) // 右耳
        );

        // 2-3次元対応点から頭部を姿勢推定する
        if (_rVec == null || _tVec == null)
        {
            _rVec = new Mat(3, 1, CvType.CV_64FC1);
            _tVec = new Mat(3, 1, CvType.CV_64FC1);
            Calib3d.solvePnP(_objectPoints, _imagePoints, _camMatrix, _distCoeffs, _rVec, _tVec);
        }

        double tVecX = _tVec.get(0, 0)[0];
        double tVecY = _tVec.get(1, 0)[0];
        double tVecZ = _tVec.get(2, 0)[0];
        bool isNotInViewport = false;
        Vector4 pos = _VP * new Vector4((float) tVecX, (float) tVecY, (float) tVecZ, 1.0f);
        if (pos.w != 0)
        {
            float x = pos.x / pos.w, y = pos.y / pos.w, z = pos.z / pos.w;
            if (x < -1.0f || x > 1.0f || y < -1.0f || y > 1.0f || z < -1.0f || z > 1.0f)
                isNotInViewport = true;
        }

        // オブジェクトがカメラ視野に存在しない場合、外部パラメータを使用しない
        if (double.IsNaN(tVecZ) || isNotInViewport)
        {
            Calib3d.solvePnP(_objectPoints, _imagePoints, _camMatrix, _distCoeffs,
                _rVec, _tVec);
        }
        else
        {
            Calib3d.solvePnP(_objectPoints, _imagePoints, _camMatrix, _distCoeffs,
                _rVec, _tVec, true, Calib3d.SOLVEPNP_ITERATIVE);
        }

        if (!isNotInViewport)
        {
            // Unityのポーズデータに変換
            double[] rVecArr = new double[3];
            _rVec.get(0, 0, rVecArr);
            double[] tVecArr = new double[3];
            _tVec.get(0, 0, tVecArr);
            PoseData poseData = ARUtils.ConvertRvecTvecToPoseData(rVecArr, tVecArr);
            // 閾値以下ならば更新を無視
            ARUtils.LowpassPoseData(ref _oldPoseData, ref poseData, positionLowPass, rotationLowPass);
            _oldPoseData = poseData;
            // 変換行列を作成
            _transformationM = Matrix4x4.TRS(poseData.pos, poseData.rot, Vector3.one);
        }


        Calib3d.Rodrigues(_rVec, _rotM);

        _transformationM.SetRow(0,
            new Vector4(
                (float) _rotM.get(0, 0)[0],
                (float) _rotM.get(0, 1)[0],
                (float) _rotM.get(0, 2)[0],
                (float) _tVec.get(0, 0)[0]));
        _transformationM.SetRow(1,
            new Vector4(
                (float) _rotM.get(1, 0)[0],
                (float) _rotM.get(1, 1)[0],
                (float) _rotM.get(1, 2)[0],
                (float) _tVec.get(1, 0)[0]));
        _transformationM.SetRow(2,
            new Vector4(
                (float) _rotM.get(2, 0)[0],
                (float) _rotM.get(2, 1)[0],
                (float) _rotM.get(2, 2)[0],
                (float) _tVec.get(2, 0)[0]));
        _transformationM.SetRow(3,
            new Vector4(
                0,
                0,
                0,
                1));

        _ARM = _invertYM * _transformationM * _invertYM;
        _ARM = _ARM * _invertYM * _invertZM;

        return ExtractRotationFromMatrix(ref _ARM).eulerAngles;
    }
    
    void SetParameter(CubismParameter parameter, float value)
    {
        if (parameter != null)
        {
            parameter.Value = Mathf.Clamp(value, parameter.MinimumValue, parameter.MaximumValue);
        }
    }


    /// <summary>
    /// 変換行列からクォータニオンを導出
    /// </summary>
    private static Quaternion ExtractRotationFromMatrix(ref Matrix4x4 matrix)
    {
        Vector3 forward;
        forward.x = matrix.m02;
        forward.y = matrix.m12;
        forward.z = matrix.m22;

        Vector3 upwards;
        upwards.x = matrix.m01;
        upwards.y = matrix.m11;
        upwards.z = matrix.m21;

        return Quaternion.LookRotation(forward, upwards);
    }
}