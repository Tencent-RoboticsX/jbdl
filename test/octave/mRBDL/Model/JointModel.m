function  [Xj,S] = JointModel( jtype, jaxis, q )

% JointModel  Calculate joint transform and motion subspace.
% [Xj,S]=JointModel(pitch,q) calculates the joint transform and motion subspace
% matrices for a revolute (pitch==0), prismatic (pitch==inf) or helical
% (pitch==any other value) joint.  For revolute and helical joints, q is
% the joint angle.  For prismatic joints, fvgfq is the linear displacement.

if jtype == 0				% revolute joint
  switch jaxis
    case 'x'
      Xj = Xrotx(q);
      S = [1;0;0;0;0;0];
    case 'y'    
      Xj = Xroty(q);
      S = [0;1;0;0;0;0];
    case 'z'    
      Xj = Xrotz(q);
      S = [0;0;1;0;0;0];
  end
else   		                % prismatic joint
  switch jaxis
    case 'x'
      Xj = Xtrans([q 0 0]);
      S = [0;0;0;1;0;0];
    case 'y'    
      Xj = Xtrans([0 q 0]);
      S = [0;0;0;0;1;0];
    case 'z'    
      Xj = Xtrans([0 0 q]);
      S = [0;0;0;0;0;1];
  end
end
