function output = sauvola(img, win_size, k_value, r_value)

if nargin < 2 || nargin > 4
  error('sauvolathresh: you must provide an input image and a window size.'); 
elseif nargin == 2
  k_value = 0.5;
  r_value = 128.0;
elseif nargin == 3
  r_value = 128.0;
end	

% Window size must be positive and odd
if win_size <= 0 || mod(win_size, 2) == 0
  error('sauvolathresh: window size must be positive and odd');
end

% Check if input image is rgb and convert to a gray-level image
if ndims(img) == 3
  img = rgb2gray(img);
end

% Convert the input image to double before doing the computations
img = double(img);

mean           = imfilter(img, fspecial('average', win_size));
stdeviation    = stdfilt(img, ones(win_size, win_size));

threshold = (mean .* (1.0 + k_value .* ((stdeviation ./ r_value) - 1.0)));

diff = threshold - img;

bw          = (diff >= 0);
output      = imcomplement(bw);
