function PlotInertiaCuboid(lxyz, pcom)
%PlotInertiaCuboid - Plot a cuboid to describe the inertia
%
% Syntax: PlotInertiaCuboid(model, q)
%
% lxyz: length along x/y/z axis
% pcom: CoM position

    org = zeros(1, 3);
    for i=1:3
        org(i) = pcom(i) - 0.5*lxyz(i);
    end
    PlotCuboid(lxyz', org, 0.1, [0, 1, 0]);

  function PlotCuboid(varargin)
      % PlotCuboid - Display a 3D-cube in the current axes
      %
      %   PlotCuboid(EDGES,ORIGIN,ALPHA,COLOR) displays a 3D-cube in the current axes
      %   with the following properties:
      %   * EDGES : 3-elements vector that defines the length of cube edges
      %   * ORIGIN: 3-elements vector that defines the start point of the cube
      %   * ALPHA : scalar that defines the transparency of the cube faces (from 0
      %             to 1)
      %   * COLOR : 3-elements vector that defines the faces color of the cube
      %
      % Example:
      %   >> PlotCuboid([5 5 5],[ 2  2  2],.8,[1 0 0]);
      %   >> PlotCuboid([5 5 5],[10 10 10],.8,[0 1 0]);
      %   >> PlotCuboid([5 5 5],[20 20 20],.8,[0 0 1]);
      
      % Default input arguments
      inArgs = { ...
        [10 56 100] , ... % Default edge sizes (x,y and z)
        [10 10  10] , ... % Default coordinates of the origin point of the cube
        .7          , ... % Default alpha value for the cube's faces
        [1 0 0]       ... % Default Color for the cube
        };
      
      % Replace default input arguments by input values
      inArgs(1:nargin) = varargin;
      
      % Create all variables
      [edges,origin,alpha,clr] = deal(inArgs{:});
      
      XYZ = { ...
        [0 0 0 0]  [0 0 1 1]  [0 1 1 0] ; ...
        [1 1 1 1]  [0 0 1 1]  [0 1 1 0] ; ...
        [0 1 1 0]  [0 0 0 0]  [0 0 1 1] ; ...
        [0 1 1 0]  [1 1 1 1]  [0 0 1 1] ; ...
        [0 1 1 0]  [0 0 1 1]  [0 0 0 0] ; ...
        [0 1 1 0]  [0 0 1 1]  [1 1 1 1]   ...
        };
      
      XYZ = mat2cell(...
        cellfun( @(x,y,z) x*y+z , ...
          XYZ , ...
          repmat(mat2cell(edges,1,[1 1 1]),6,1) , ...
          repmat(mat2cell(origin,1,[1 1 1]),6,1) , ...
          'UniformOutput',false), ...
        6,[1 1 1]);
      
      
      cellfun(@patch,XYZ{1},XYZ{2},XYZ{3},...
        repmat({clr},6,1),...
        repmat({'FaceAlpha'},6,1),...
        repmat({alpha},6,1)...
        );
      
      view(3);
  end
end