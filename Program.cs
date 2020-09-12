using System;
using System.Drawing;
using OpenCVMarkerLessAR;
using OpenCvSharp;
namespace MarkerLessDetector
{
    class Program
    {
        static void Main(string[] args)
        {
			var cap = new VideoCapture(@"rabit3V.mp4");
			Mat MARKER = new Mat("rabits.jpg");
			Cv2.Resize(MARKER, MARKER, new OpenCvSharp.Size(500, 500));
			Pattern pattern = new Pattern();
			PatternTrackingInfo patternTrackinginfo = new PatternTrackingInfo();
			PatternDetector detector = new PatternDetector(true);
			detector.buildPatternFromImage(MARKER, pattern);
			detector.train(pattern);



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

					}


					window.ShowImage(img);
					Cv2.WaitKey(sleepTime);
					img.Release();
				}
			}
			cap.Release();
		}
    }
}
