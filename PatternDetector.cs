using System;
using System.Collections.Generic;
using System.Linq;
//using OpenCVForUnity.CoreModule;
//using OpenCVForUnity.Features2dModule;
//using OpenCVForUnity.ImgprocModule;
//using OpenCVForUnity.Calib3dModule;
using OpenCvSharp;

namespace OpenCVMarkerLessAR
{
    /// <summary>
    /// Pattern detector.
    /// This code is a rewrite of https://github.com/MasteringOpenCV/code/tree/master/Chapter3_MarkerlessAR using "OpenCV for Unity".
    /// </summary>
    public class PatternDetector
    {
        /// <summary>
        /// The enable ratio test.
        /// </summary>
        public bool enableRatioTest;

        /// <summary>
        /// The enable homography refinement.
        /// </summary>
        public bool enableHomographyRefinement;

        /// <summary>
        /// The homography reprojection threshold.
        /// </summary>
        public float homographyReprojectionThreshold;

        /// <summary>
        /// The m_query keypoints.
        /// </summary>
        //MatOfKeyPoint m_queryKeypoints;

        KeyPoint[] m_queryKeypoints;

        /// <summary>
        /// The m_query descriptors.
        /// </summary>
        Mat m_queryDescriptors;

        /// <summary>
        /// The m_matches.
        /// </summary>
        DMatch[] m_matches;
        //MatOfDMatch m_matches;

        /// <summary>
        /// The m_knn matches.
        /// </summary>
        /// List<MatOfDMatch> m_knnMatches
        List<DMatch[]> m_knnMatches;
        //Mat m_knnMatches;
        /// <summary>
        /// The m_gray image.
        /// </summary>
        Mat m_grayImg;

        /// <summary>
        /// The m_warped image.
        /// </summary>
        Mat m_warpedImg;

        /// <summary>
        /// The m_rough homography.
        /// </summary>
        Mat m_roughHomography;

        /// <summary>
        /// The m_refined homography.
        /// </summary>
        Mat m_refinedHomography;

        /// <summary>
        /// The m_pattern.
        /// </summary>
        Pattern m_pattern;

        /// <summary>
        /// The m_detector.
        /// </summary>
        ORB m_detector;

        /// <summary>
        /// The m_extractor.
        /// </summary>
        ORB m_extractor;

        /// <summary>
        /// The m_matcher.
        /// </summary>
        //DescriptorMatcher m_matcher;
        DescriptorMatcher m_matcher;

        /// <summary>
        /// Initializes a new instance of the <see cref="PatternDetector"/> class.
        /// </summary>
        /// <param name="detector">Detector.</param>
        /// <param name="extractor">Extractor.</param>
        /// <param name="matcher">Matcher.</param>
        /// <param name="ratioTest">If set to <c>true</c> ratio test.</param>
        /// 
        public  Point2d Point2fToPoint2d(Point2f pf) => new Point2d(((int)pf.X), ((int)pf.Y));
        public PatternDetector (bool ratioTest)
        {

            m_detector = ORB.Create(1000);
           
            m_extractor = ORB.Create(1000);
            //BFMatcher bfMatcher = new BFMatcher(NormTypes.Hamming, true);
            m_matcher = new BFMatcher(NormTypes.Hamming);
            //m_matcher = DescriptorMatcher.Create("BRUTEFORCE_HAMMING");
            enableRatioTest = ratioTest;
            enableHomographyRefinement = true;
            homographyReprojectionThreshold = 3;

            //m_queryKeypoints = new MatOfKeyPoint ();
            m_queryDescriptors = new Mat ();
            //m_matches = new MatOfDMatch ();
            //m_knnMatches = new List<MatOfDMatch> ();
            m_grayImg = new Mat ();
            m_warpedImg = new Mat ();
            m_roughHomography = new Mat ();
            m_refinedHomography = new Mat ();

        }

        /// <summary>
        /// Train the specified pattern.
        /// </summary>
        /// <param name="pattern">Pattern.</param>
        public void train ()
        {
            // Store the pattern object

        
            // API of cv::DescriptorMatcher is somewhat tricky
            // First we clear old train data:
            m_matcher.Clear();
        
            // Then we add vector of descriptors (each descriptors matrix describe one image). 
            // This allows us to perform search across multiple images:
            List<Mat> descriptors = new List<Mat> (1);
            descriptors.Add (m_pattern.descriptors.Clone ()); 
            m_matcher.Add (descriptors);
        
            // After adding train data perform actual train:
            m_matcher.Train ();
        }

        /// <summary>
        /// Builds the pattern from image.
        /// </summary>
        /// <param name="image">Image.</param>
        /// <param name="pattern">Pattern.</param>
        public void buildPatternFromImage (Mat image, Pattern pattern)
        {
            //        int numImages = 4;
            //        float step = Mathf.Sqrt (2.0f);

            // Store original image in pattern structure
            // Image dimensions
            m_pattern = pattern;
            float w = image.Cols;
            float h = image.Rows;
            m_pattern.size = new Size (w, h);
            m_pattern.frame = image.Clone ();
            //getGray (image, pattern.grayImg);
            Cv2.CvtColor(image, m_pattern.grayImg, ColorConversionCodes.BGR2GRAY);

            //pattern.points2d.fromList (points2dList);
            m_pattern.points2d = new Point2f[]
            {
                new Point2f(0, 0),
                new Point2f(w, 0),
                new Point2f(w,h),
                new Point2f(0,h)
            };


            m_pattern.points3d = new Point3f[]
            {
                //18 mean marker cm
                new Point3f(-8, -8, 0),
                new Point3f(8, -8, 0),
                new Point3f(8, 8, 0),
                new Point3f(-8, 8, 0)
            };


            extractFeatures (pattern.grayImg,ref pattern.keyPoints,ref pattern.descriptors);
        }

        /// <summary>
        /// Finds the pattern.
        /// </summary>
        /// <returns><c>true</c>, if pattern was found, <c>false</c> otherwise.</returns>
        /// <param name="image">Image.</param>
        /// <param name="info">Info.</param>
        public bool findPattern (Mat image, PatternTrackingInfo info)
        {
            // Convert input image to gray
            getGray (image,ref m_grayImg);
            //Cv2.CvtColor(image, m_grayImg, ColorConversionCodes.BGR2GRAY);
            // Extract feature points from input gray image
            extractFeatures (m_grayImg, ref m_queryKeypoints, ref m_queryDescriptors);
        
            // Get matches with current pattern
            getMatches (m_queryDescriptors,ref m_matches);

            m_roughHomography = new Mat();
            // Find homography transformation and detect good matches
            bool homographyFound = refineMatchesWithHomography (
                                       m_queryKeypoints, 
                                       m_pattern.keyPoints, 
                                       homographyReprojectionThreshold, 
                                       ref m_matches, 
                                       ref m_roughHomography,
                                       HomographyMethods.Ransac);
        
            if (homographyFound) {
                        

                        
                // If homography refinement enabled improve found transformation
                if (enableRatioTest) {
                    // Warp image using found homography
                    Cv2.WarpPerspective(m_grayImg, m_warpedImg, m_roughHomography, m_pattern.size,InterpolationFlags.WarpInverseMap| InterpolationFlags.Cubic);
                    Cv2.ImShow("second img", m_warpedImg);
                    KeyPoint[] warpedqueryKeypoints=null;
                    extractFeatures(m_warpedImg, ref warpedqueryKeypoints, ref m_queryDescriptors);
                    DMatch[] refinedMatches = null;
                    // Match with pattern
                    getMatches(m_queryDescriptors, ref refinedMatches);
                    m_refinedHomography = new Mat();
                    // Estimate new refinement homography
                    homographyFound = refineMatchesWithHomography(
                        warpedqueryKeypoints,
                        m_pattern.keyPoints,
                        homographyReprojectionThreshold,
                        ref refinedMatches,
                        ref m_refinedHomography,
                        HomographyMethods.LMedS);
                    if (!homographyFound)
                        return false;
                    info.homography = m_roughHomography * m_refinedHomography;
                    Cv2.WarpPerspective(m_grayImg, m_warpedImg, info.homography, m_pattern.size, InterpolationFlags.WarpInverseMap | InterpolationFlags.Cubic);
                    Cv2.ImShow("third img", m_warpedImg);
                    Console.WriteLine(m_pattern.points2d.Length);
                    using (Mat src = new Mat(m_pattern.points2d.Length, 1, MatType.CV_32FC2, m_pattern.points2d))
                    using (Mat dst = new Mat())
                    {
                        Cv2.PerspectiveTransform(src, dst, info.homography);
                        Point2f[] dstArray = new Point2f[dst.Rows * dst.Cols];
                        dst.GetArray(out dstArray);
                        for (int j = 0; j < dstArray.Length; j++)
                            Console.WriteLine(dstArray[j]);
                        Point2d[]result = Array.ConvertAll(dstArray,new Converter<Point2f, Point2d>(Point2fToPoint2d));
                        for (int j = 0; j < result.Length; j++)
                            Console.WriteLine(result[j]);
                        info.points2d = new Mat(result.Length, 1, MatType.CV_32FC2, result);
                        //for (int j = 0; j < info.points2d.Rows; j++)
                        //    Console.WriteLine(info.points2d.Row(j));
                        var s = "break point";
                        //return result;
                    }
                    
                    



                } else {
                    info.homography = m_roughHomography;

                    using (Mat src = new Mat(m_pattern.points2d.Length, 1, MatType.CV_32FC2, m_pattern.points2d))
                    using (Mat dst = new Mat())
                    {
                        Cv2.PerspectiveTransform(src, dst, info.homography);
                        Point2f[] dstArray = new Point2f[dst.Rows * dst.Cols];
                        dst.GetArray(out dstArray);
                        Point2d[] result = Array.ConvertAll(dstArray, new Converter<Point2f, Point2d>(Point2fToPoint2d));
                        info.points2d = new Mat(result.Length, 1, MatType.CV_32FC2, result);
                    }

                   
                                
                }
            }


        
            return homographyFound;
        }

        /// <summary>
        /// Gets the gray.
        /// </summary>
        /// <param name="image">Image.</param>
        /// <param name="gray">Gray.</param>
        static void getGray (Mat image,ref Mat gray)
        {

            Mat labImage = new Mat();
            Mat grayImage = new Mat();
            Mat binaryImage = new Mat();
            Mat edgeImage = new Mat();

            int w = image.Cols, h = image.Rows;

            // 以 L 二值化，並做邊緣檢測
            Cv2.CvtColor(image, labImage, ColorConversionCodes.BGR2Lab);
            Mat[] labChannel = Cv2.Split(labImage); //分割通道
            Mat[] grayChannel = new Mat[] { labChannel[0] };
            Cv2.MixChannels(labChannel, grayChannel, new int[] { 0, 0 });
            Cv2.Merge(grayChannel, gray);

                        
        }

        /// <summary>
        /// Extracts the features.
        /// </summary>
        /// <returns><c>true</c>, if features was extracted, <c>false</c> otherwise.</returns>
        /// <param name="image">Image.</param>
        /// <param name="keypoints">Keypoints.</param>
        /// <param name="descriptors">Descriptors.</param>
        bool extractFeatures (Mat image,ref KeyPoint[] keypoints, ref Mat descriptors)
        {
            if (image.Total () == 0) {
                return false;
            }
            if (image.Channels () != 1) {
                return false;
            }
            keypoints=m_detector.Detect (image, null);
            if (keypoints.Length == 0)
                return false;
            m_extractor.Compute(image,ref keypoints, descriptors);
            //m_detector.DetectAndCompute(image, ref keypoints, descriptors);
           // m_extractor.compute (image, keypoints, descriptors);
            if (keypoints.Length == 0)
                return false;

        
            return true;
        }

        /// <summary>
        /// Gets the matches.
        /// </summary>
        /// <param name="queryDescriptors">Query descriptors.</param>
        /// <param name="matches">Matches.</param>
        void getMatches (Mat queryDescriptors, ref DMatch[] matches)
        {

            List<DMatch> matchesList = new List<DMatch> ();
//      matches.clear();
        
            if (enableRatioTest)
            {
                // To avoid NaN's when best match has zero distance we will use inversed ratio. 
                float minRatio = 1.0f / 1.5f;

                // KNN match will return 2 nearest matches for each query descriptor
                m_knnMatches=m_matcher.KnnMatch (queryDescriptors, 2).ToList();
                //m_matcher.KnnMatch()
                for (int i = 0; i < m_knnMatches.Count; i++) {
                    DMatch bestMatch = m_knnMatches[i][0];
                    DMatch betterMatch = m_knnMatches[i][1];
                
                    float distanceRatio = bestMatch.Distance / betterMatch.Distance;
                
                    // Pass only matches where distance ratio between 
                    // nearest matches is greater than 1.5 (distinct criteria)
                    if (distanceRatio < minRatio) {

                        matchesList.Add (bestMatch);
                    }
                }

                matches=matchesList.ToArray();

            } else {
                // Perform regular match
                matches=m_matcher.Match (queryDescriptors);

            }

//        Debug.Log ("getMatches " + matches.ToString ());
        }

        /// <summary>
        /// Refines the matches with homography.
        /// </summary>
        /// <returns><c>true</c>, if matches with homography was refined, <c>false</c> otherwise.</returns>
        /// <param name="queryKeypoints">Query keypoints.</param>
        /// <param name="trainKeypoints">Train keypoints.</param>
        /// <param name="reprojectionThreshold">Reprojection threshold.</param>
        /// <param name="matches">Matches.</param>
        /// <param name="homography">Homography.</param>
        static bool refineMatchesWithHomography
        (
            KeyPoint[] queryKeypoints,
            KeyPoint[] trainKeypoints, 
            float reprojectionThreshold,
            ref DMatch[] matches,
            ref Mat homography,
            HomographyMethods methods
        )
        {
//              Debug.Log ("matches " + matches.ToString ());

            int minNumberMatchesAllowed = 8;

            List<KeyPoint> queryKeypointsList = queryKeypoints.ToList();
            List<KeyPoint> trainKeypointsList = trainKeypoints.ToList ();
            List<DMatch> matchesList = matches.ToList ();
            
            if (matchesList.Count < minNumberMatchesAllowed)
                return false;
        
            // Prepare data for cv::findHomography
            List<Point2f> srcPointsList = new List<Point2f> (matchesList.Count);
            List<Point2f> dstPointsList = new List<Point2f> (matchesList.Count);
        
            for (int i = 0; i < matchesList.Count; i++) {
                srcPointsList.Add (trainKeypointsList [matchesList [i].TrainIdx].Pt);
                dstPointsList.Add (queryKeypointsList [matchesList [i].QueryIdx].Pt);
            }
            // Find homography matrix and get inliers mask


            //              Debug.Log ("srcPoints " + srcPoints.ToString ());
            //              Debug.Log ("dstPoints " + dstPoints.ToString ());
            using(Mat srcPoints =new Mat(srcPointsList.Count,2,MatType.CV_32F, srcPointsList.ToArray()))
            using (Mat dstPoints = new Mat(dstPointsList.Count, 2, MatType.CV_32F, dstPointsList.ToArray()))
            using (var inliersMask = new Mat(srcPointsList.Count, 1, MatType.CV_8U))
            {
                //homography=Cv2.FindHomography(InputArray.Create( srcPointsList), InputArray.Create(dstPointsList), HomographyMethods.Ransac, reprojectionThreshold, inliersMask);
                homography = Cv2.FindHomography(srcPoints, dstPoints, methods, reprojectionThreshold, inliersMask);
                if (homography.Rows != 3 || homography.Cols != 3)
                    return false;


                List<DMatch> inliers = new List<DMatch>();
                for (int i = 0; i < inliersMask.Rows; i++)
                {
                    if (inliersMask.Get<byte>(i,0) == 1)
                        inliers.Add(matchesList[i]);
                }
                matches = inliers.ToArray();
            }





            return matchesList.Count > minNumberMatchesAllowed;
        }
    }
}