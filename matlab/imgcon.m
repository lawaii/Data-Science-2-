function results = myimfcn(varargin)
%Image Processing Function
%
% VARARGIN - Can contain up to two inputs: 
%   IM - First input is a numeric array containing the image data. 
%   INFO - Second input is a scalar structure containing information about 
%          the input image source.
%
%   INFO can be used to obtain metadata about the image read. 
%   To apply a batch function using the INFO argument, you must select the 
%   Include Image Info check box in the app toolstrip.
%   
% RESULTS - A scalar struct with the processing results.
%
% 
%
%--------------------------------------------------------------------------
% Auto-generated by imageBatchProcessor App. 
%
% When used by the App, this function will be called for each input image
% file automatically.
%
%--------------------------------------------------------------------------

% Input parsing------------------------------------------------------------
im = varargin{1};

if nargin == 2
    % Obtain information about the input image source
    info = varargin{2};
end

% Replace the sample below with your code----------------------------------

imcon = imadjust(im,[0 1],[0.3 0.6]);

results.imcon = imcon;

%--------------------------------------------------------------------------
