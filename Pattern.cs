//using OpenCVForUnity.CoreModule;
using OpenCvSharp;
namespace OpenCVMarkerLessAR
{
    /// <summary>
    /// Pattern.
    /// This code is a rewrite of https://github.com/MasteringOpenCV/code/tree/master/Chapter3_MarkerlessAR using "OpenCV for Unity".
    /// </summary>
    public class Pattern
    {
        /// <summary>
        /// The size.
        /// </summary>
        public Size size;

        /// <summary>
        /// The frame.
        /// </summary>
        public Mat frame;

        /// <summary>
        /// The gray image.
        /// </summary>
        public Mat grayImg;

        /// <summary>
        /// The keypoints.
        /// </summary>
        /// 
        public KeyPoint[] keyPoints;
        //public MatOfKeyPoint keypoints;

        /// <summary>
        /// The descriptors.
        /// </summary>
        public Mat descriptors;

        /// <summary>
        /// The points2d.
        /// </summary>
        public Point2f[] points2d;// map to 236行
        //public MatOfPoint2f points2d;

        /// <summary>
        /// The points3d.
        /// </summary>
        /// 
        public Point3f[] points3d;
        //public MatOfPoint3f points3d;

        /// <summary>
        /// Initializes a new instance of the <see cref="Pattern"/> class.
        /// </summary>
        /// 
        //public Mat points2d;
        //w.Set<Point2d>(0, new Point2d(0, 0));
        //            w.Set<Point2d>(1, new Point2d(0, 0));
        //            w.Set<Point2d>(2, new Point2d(0, 0));
        //            w.Set<Point2d>(3, new Point2d(0, 0));
        public Pattern ()
        {
            size = new Size ();
            frame = new Mat ();
            grayImg = new Mat ();
            //keypoints = new MatOfKeyPoint ();
            descriptors = new Mat ();

        }
    }
}