//using UnityEngine;
using System.Collections.Generic;
using OpenCvSharp;
using OpenTK;
//using OpenCVForUnity.CoreModule;
//using OpenCVForUnity.Calib3dModule;
//using OpenCVForUnity.ImgprocModule;

namespace OpenCVMarkerLessAR
{
    /// <summary>
    /// Pattern tracking info.
    /// This code is a rewrite of https://github.com/MasteringOpenCV/code/tree/master/Chapter3_MarkerlessAR using "OpenCV for Unity".
    /// </summary>
    public class PatternTrackingInfo
    {
        /// <summary>
        /// The homography.
        /// </summary>
        public Mat homography;

        /// <summary>
        /// The points2d.
        /// </summary>
        public Mat points2d;//Point2f[]
        //public MatOfPoint2f points2d;

        /// <summary>
        /// The pose3d.
        /// </summary>
        public Matrix4 pose3d;
        //public Matrix4x4 pose3d;


        public Mat campos;
        /// <summary>
        /// The cameraMatrix.
        /// </summary>

        private double[,] cameraMatrix = new double[3, 3]
        {
            { 2.39175513e+03, 0, 6.47489545e+02 },
            { 0, 2.43199146e+03, 5.50328902e+02 },
            { 0, 0, 1 }
        };
        //-4.64533892e+00,8.57754698e+02,-2.09117033e-02,2.64844459e-02,- 3.72599688e+04

        double[] dist = new double[] { -4.64533892e+00, 8.57754698e+02, -2.09117033e-02, 2.64844459e-02, -3.72599688e+04 };
        /// <summary>
        /// Initializes a new instance of the <see cref="PatternTrackingInfo"/> class.
        /// </summary>
        public PatternTrackingInfo ()
        {
            homography = new Mat ();
            points2d = new Mat();
            campos = new Mat();
            //points2d = new MatOfPoint2f ();
            //pose3d = new Matrix4x4 ();
        }

        /// <summary>
        /// Computes the pose.
        /// </summary>
        /// <param name="pattern">Pattern.</param>
        /// <param name="camMatrix">Cam matrix.</param>
        /// <param name="distCoeff">Dist coeff.</param>
        public void computePose (Pattern pattern)
        {
            //Mat Rvec = new Mat ();
            //Mat Tvec = new Mat ();
            Mat raux = new Mat ();
            Mat taux = new Mat ();



            //Mat temp1 = pattern.points3d;
            using (Mat points3dsrc = new Mat(pattern.points3d.Length, 3, MatType.CV_64F, pattern.points3d))
            {

                Cv2.SolvePnP(points3dsrc, points2d,
                    new Mat(3, 3, MatType.CV_64F, cameraMatrix),
                    new Mat(5, 1, MatType.CV_64F, dist),
                    raux, taux);
            }



            Mat rotMat = new Mat();

            Cv2.Rodrigues(raux, rotMat);
            campos = -rotMat.T() * taux;

            var a = campos.Get<Point3d>(0).ToString();
            //Mat rotMat = new Mat (3, 3, CvType.CV_64FC1); 
            //Calib3d.Rodrigues (Rvec, rotMat);

            pose3d = new Matrix4
            {
                Row0 = new Vector4(rotMat.Get<float>(0, 0), rotMat.Get<float>(0, 1), rotMat.Get<float>(0, 2), taux.Get<float>(0, 0)),
                Row1 = new Vector4(rotMat.Get<float>(1, 0), rotMat.Get<float>(1, 1), rotMat.Get<float>(1, 2), taux.Get<float>(1, 0)),
                Row2 = new Vector4(rotMat.Get<float>(2, 0), rotMat.Get<float>(2, 1), rotMat.Get<float>(2, 2), taux.Get<float>(2, 0)),
                Row3 = new Vector4(0, 0, 0, 1.0f),
            };


            //      Debug.Log ("pose3d " + pose3d.ToString ());



            raux.Dispose ();
            taux.Dispose ();
            rotMat.Dispose ();
        }

        /// <summary>
        /// Draw2ds the contour.
        /// </summary>
        /// <param name="image">Image.</param>
        /// <param name="color">Color.</param>
        public void draw2dContour(Mat image, Scalar color)
        {
            //      Debug.Log ("points2d " + points2d.dump());

            //List<OpenCvSharp.Point> points2dList = points2d.toList();
            int n = points2d.Rows;
            for (int i = 0; i < n; i++)
            {
                Cv2.Line(image, (int)points2d.Get<Point2d>(i).X, (int)points2d.Get<Point2d>(i).Y, (int)points2d.Get<Point2d>((i+1)%n).X, (int)points2d.Get<Point2d>((i + 1) % n).Y, color, 50, LineTypes.AntiAlias, 0);
                ///Cv2.Line(image, 10,20,30,40, color, 50, LineTypes.AntiAlias, 0);
            }
        }
    }
}