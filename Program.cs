using System;
using System.Drawing;
using OpenCVMarkerLessAR;
using OpenCvSharp;
namespace MarkerLessDetector
{
    class Program
    {
		static void VideoRun()
        {

			var cap = new VideoCapture(@"rabit3V.mp4");
			Mat MARKER = new Mat("rabits.jpg");
			Cv2.Resize(MARKER, MARKER, new OpenCvSharp.Size(500, 500));
			Pattern pattern = new Pattern();
			PatternTrackingInfo patternTrackinginfo = new PatternTrackingInfo();
			PatternDetector detector = new PatternDetector(true);
			detector.buildPatternFromImage(MARKER, pattern);
			detector.train();



			int sleepTime = 1;
			using (Window window = new Window("capture"))
			using (Mat image = new Mat())
			{
				while (true)
				{
					cap.Read(image);
					if (image.Empty())
						break;
					var img = image.Clone();

					if (detector.findPattern(img, patternTrackinginfo))
					{
                        patternTrackinginfo.computePose(pattern);

                        var temp = patternTrackinginfo.campos.Get<Point3d>(0);

                        string camposInfo = "x:" + Math.Round(temp.X, 5) + "\ny:" + Math.Round(temp.Y, 5) + "\nz:" + Math.Round(temp.Z, 5);
                        Cv2.PutText(img,
                        camposInfo,
                        new OpenCvSharp.Point(0, 80),
                        HersheyFonts.HersheyComplex,
                        0.5,
                        Scalar.White);

                        for (int i = 0; i < patternTrackinginfo.points2d.Rows; i++)
						{
							//Console.WriteLine(" x"+(int)patternTrackinginfo.points2d.Get<Point2d>(i).X+" "+ (int)patternTrackinginfo.points2d.Get<Point2d>(i).Y);
							Cv2.Circle(img, (int)patternTrackinginfo.points2d.Get<Point2d>(i).X, (int)patternTrackinginfo.points2d.Get<Point2d>(i).Y, 5, Scalar.Black, 3);
						}
					}


					window.ShowImage(img);
					Cv2.WaitKey(sleepTime);
					img.Release();
				}
			}
			cap.Release();
		}
		static void imgRun()
        {
			string path = @"D:\Code_Resource\IMAGE\";
			Mat img = new Mat(path+ "rabit3.jpg");
			Mat MARKER = new Mat(path+ "rabits.jpg");
			Cv2.Resize(MARKER, MARKER, new OpenCvSharp.Size(500, 500));
			Pattern pattern = new Pattern();
			PatternTrackingInfo patternTrackinginfo = new PatternTrackingInfo();
			PatternDetector detector = new PatternDetector(true);
			detector.buildPatternFromImage(MARKER, pattern);
			detector.train();
			if(detector.findPattern(img, patternTrackinginfo))
            {
				patternTrackinginfo.computePose(pattern);

				var temp = patternTrackinginfo.campos.Get<Point3d>(0);

				string camposInfo = "x:" + Math.Round(temp.X, 5) + "y:" + Math.Round(temp.Y, 5) + "z:" + Math.Round(temp.Z, 5);
				Cv2.PutText(img,
				camposInfo,
				new OpenCvSharp.Point(0, 80),
				HersheyFonts.HersheyComplex,
				0.5,
				Scalar.White);

				for (int i = 0; i < 4; i++)
				{
					Cv2.Circle(img, (int)patternTrackinginfo.points2d.Get<Point2d>(i).X, (int)patternTrackinginfo.points2d.Get<Point2d>(i).Y, 5, Scalar.Black, 3);
				}
			}
			Cv2.ImShow("result", img);

			Cv2.WaitKey(100000);
			img.Release();
		}
        static void Main(string[] args)
        {
			//imgRun();
			VideoRun();


		}
    }
}
