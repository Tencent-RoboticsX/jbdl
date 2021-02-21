function X_T = SpatialTransform(E,r)
%SPATIALMATRIX Summary of this function goes here
%   Detailed explanation goes here
X_T = [E, zeros(3);
       -E*CrossMatrix(r), E];
end

