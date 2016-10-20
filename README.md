# curvePieceEstimate
Curve Piece Estimate (SLAM). Attempt to construct a curved trajectory estimate from point cloud data.

'''
Curve Piece Estimate

Attempt to construct a curved trajectory estimate from a point cloud of features. (This is SLAM.)
Different feature types may be used, but one is used at the moment.
- discontinuity features (dif)

If you use this code, or it helps you in any way, please cite:

Lehtola, V. V., Virtanen, J. P., Vaaja, M. T., Hyyppä, H., & Nüchter, A. (2016). 
Localization of a mobile laser scanner via dimensional reduction. 
ISPRS Journal of Photogrammetry and Remote Sensing, 121, 48-59.

Also if you use intrinsic localization (3D point clouds using only one laser scanner/profiler), please cite

Lehtola, V. V., Virtanen, J. P., Kukko, A., Kaartinen, H., & Hyyppä, H. (2015). 
Localization of mobile laser scanner using classical mechanics. 
ISPRS Journal of Photogrammetry and Remote Sensing, 99, 25-29. 

Uses 3DTK format, http://slam6d.sourceforge.net/

Reads .feature files.
Writes .pose and .frames files.

See example.

Created on Jul 1, 2015

@author: Ville Lehtola, ville.lehtola@iki.fi

'''
