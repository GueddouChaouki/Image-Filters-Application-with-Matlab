function varargout = Projet_menu(varargin)
% PROJET_MENU MATLAB code for Projet_menu.fig
%      PROJET_MENU, by itself, creates a new PROJET_MENU or raises the existing
%      singleton*.
%
%      H = PROJET_MENU returns the handle to a new PROJET_MENU or the handle to
%      the existing singleton*.
%
%      PROJET_MENU('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PROJET_MENU.M with the given input arguments.
%
%      PROJET_MENU('Property','Value',...) creates a new PROJET_MENU or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Projet_menu_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Projet_menu_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Projet_menu

% Last Modified by GUIDE v2.5 06-May-2017 06:53:25

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Projet_menu_OpeningFcn, ...
                   'gui_OutputFcn',  @Projet_menu_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Projet_menu is made visible.
function Projet_menu_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Projet_menu (see VARARGIN)

% Choose default command line output for Projet_menu
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Projet_menu wait for user response (see UIRESUME)
% uiwait(handles.figure1);


         

% --- Outputs from this function are returned to the command line.
function varargout = Projet_menu_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



% --------------------------------------------------------------------
function correction_gamma_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
img=imread('rice2.tif');
img1=imadjust(img,[],[],1);
img2=imadjust(img,[],[],7);
img3=imadjust(img,[],[],0.1);
subplot(2,2,1, 'Parent', hp);imshow(img);title('image originale');
subplot(2,2,2, 'Parent', hp);imshow(img1);title('gamma=1');
subplot(2,2,3, 'Parent', hp);imshow(img2);title('gamma>1'); % sombre
subplot(2,2,4, 'Parent', hp);imshow(img3);title('gamma<1'); % claire

% --------------------------------------------------------------------
function Inversion_Dynamique_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
J= imread('rice2.tif');
invert = 255 - J;
a1= imhist(J);
a2= imhist(invert);
subplot(2,1,1, 'Parent', hp);imshow(J);title('image original');
subplot(2,1,2, 'Parent', hp);imshow(invert);title('image inverse');

% --------------------------------------------------------------------
function Transf_logaritmique_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
g=imread('cameraman.tif');
c = 0.5;
[M,N]=size(g);
        for x = 1:M
            for y = 1:N  
               z(x,y)=c.*log10(1+double(g(x,y))); 
              
            end
        end
subplot(1,2,1, 'Parent', hp);imshow(g);title('image avant');
subplot(1,2,2, 'Parent', hp);imshow(z);title('image apr?s');

% --------------------------------------------------------------------
function Kmeans_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
warning off
[x,map] = dicomread('MR-MONO2-16-knee');
imshow(x,map);
info = dicominfo('MR-MONO2-16-knee');
info
a = rgb2gray(imread('MR-MONO2-16-knee.jpg'));
imshow(a);
imdata = reshape(a,[],1);
imdata = double(imdata);
[IDX nn] = kmeans(imdata,4);
imIDX = reshape(IDX,size(a));
subplot(2,3,1, 'Parent', hp);imshow(a);title('Image originale');
subplot(2,3,2, 'Parent', hp);imshow(imIDX,[]);title('k-means result');
subplot(2,3,3, 'Parent', hp);imshow(imIDX==1,[]);title('cluster1');
subplot(2,3,4, 'Parent', hp);imshow(imIDX==2,[]);title('cluster2');
subplot(2,3,5, 'Parent', hp);imshow(imIDX==3,[]);title('cluster3');
subplot(2,3,6, 'Parent', hp);imshow(imIDX==4,[]);title('cluster4');

% --------------------------------------------------------------------
function Quadtree_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
X=imread('coins.png');
X = mat2gray(X);
taille_min =2;
[blocks,g]=splitmerge(X,taille_min,@predicate);
subplot(2,1,1, 'Parent', hp);imshow([blocks,g]);title('max(max(region))-min(min(region))>0.2');
[blocks,g]=splitmerge(X,taille_min,@predicate2);
subplot(2,1,2, 'Parent', hp);imshow([blocks,g]);title('std2(region)>0.05');

% --------------------------------------------------------------------
function Sobel_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
             'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);

I=imread('cameraman.tif');
J=I;%rgb2gray (I);
J=double(J)/255.0;
seuil=0.8;
H=fspecial('sobel');
V=-H';
Gh=filter2(H,J);
Gv=filter2(V,J);
G=sqrt(Gh.*Gh + Gv.*Gv);
Gs1=(G>seuil*4/3);
subplot(2,2,1, 'Parent', hp);imshow(Gh);
subplot(2,2,2, 'Parent', hp);imshow(Gv);
subplot(2,2,3, 'Parent', hp);imshow(G);
subplot(2,2,4, 'Parent', hp);imshow(Gs1);
% --------------------------------------------------------------------
function canni_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
             'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
img = imread('rice2.tif');
  cont2 = edge(img,'canny');
  subplot(1,2,1, 'Parent', hp);imshow(img);title('Image original');
  subplot(1,2,2, 'Parent', hp),imshow(cont2); title('canni');
  
% --------------------------------------------------------------------
function prewitt_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
             'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
img = imread('rice2.tif');
  cont2 = edge(img,'prewitt');
  subplot(1,2,1, 'Parent', hp);imshow(img);title('Image original');
  subplot(1,2,2, 'Parent', hp),imshow(cont2); title('prewitt');


% --------------------------------------------------------------------
function morphologique_Callback(hObject, eventdata, handles) 
hp = uipanel('Title','Main Panel','FontSize',12,...
             'BackgroundColor','white',...
             'Position',[.20 .1 .78 .87]);
Image= imread('coins.png');
imshow(Image);

h=Image;
s= strel('disk',5);  % square
i=imerode(h,s);
j=imdilate(h,s);
subplot(1,3,1, 'Parent', hp); imshow(h);title('image originale');
subplot(1,3,2, 'Parent', hp); imshow(i);title('image erroder');
subplot(1,3,3, 'Parent', hp); imshow(j);title('image delater');

% --------------------------------------------------------------------
function Median_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
             'BackgroundColor','white',...
             'Position',[.20 .1 .78 .87]);
X1=imread('cameraman.tif')
X3=imnoise(X1,'salt & pepper',0.05)
Y3=medfilt2(X3,[3,3])     % filtre m?dian
subplot(1,3,1, 'Parent', hp);imshow(X1);title('Image original');
subplot(1,3,2, 'Parent', hp); imshow(X3); title('un bruit poivre et sel');
subplot(1,3,3, 'Parent', hp),imshow(Y3); title('filtre m?dian');

% --------------------------------------------------------------------
function Robert_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
           'Position',[.20 .1 .78 .87]);
    % create a new figure to show the image . 
    newImg = imread('lena_color.tif');
    % convert RGB to gray scale.
    grayImage= rgb2gray(newImg);
    % apply roberts filter.
    robertsResult = edge(grayImage,'roberts');

subplot(1,2,1, 'Parent', hp);imshow(newImg); title('Image originale');
subplot(1,2,2, 'Parent', hp);imshow(robertsResult); title('roberts Result');



% --------------------------------------------------------------------
function moyenneur_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
           'Position',[.20 .1 .78 .87]);
X1=imread('cameraman.tif')
X2= imnoise(X1,'gaussian',0.01) % bruit blanc Gaussien
h = fspecial('average',[3 3])% un filtre moyenneur de taille 3 ? 3
Y2=imfilter(X2,h,'replicate') 
subplot(1,3,1, 'Parent', hp);imshow(X1);title('Image original');
subplot(1,3,2, 'Parent', hp); imshow(X2); title('un bruit Gaussien');
subplot(1,3,3, 'Parent', hp),imshow(Y2); title('filtre moyenneur');

% --------------------------------------------------------------------
function gaussian_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
           'Position',[.20 .1 .78 .87]);
X1= imread('cameraman.tif');
X2= imnoise(X1,'gaussian',0.01) % bruit blanc Gaussien
X3=imnoise(X1,'salt & pepper',0.05)

h = fspecial('gaussian',[15 15],1.5)
Y2=imfilter(X2,h,'replicate')
Y3=imfilter(X3,h,'replicate')

subplot(1,3,1, 'Parent', hp);imshow(X1);title('Image original');
subplot(1,3,2, 'Parent', hp); imshow(X3); title('un bruit poivre et sel');
subplot(1,3,3, 'Parent', hp),imshow(Y3); title('filtre Gaussien');

% --------------------------------------------------------------------
function ouvrir_Callback(hObject, eventdata, handles)
 [filename pathname] = uigetfile({'*.jpg';'*.bmp'},'File Selector');
 handles.myImage = strcat(pathname, filename);
 axes(handles.axesImage);
 Image= handles.myImage;
 imshow(Image)
 % save the updated handles object
 guidata(hObject,handles);
 
% --------------------------------------------------------------------
function Enregistrer_Callback(hObject, eventdata, handles)
% hObject    handle to Enregistrer (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
F = getframe(handles.axesImage);
Image = frame2im(F);
imwrite(Image, 'Image.jpg')

% --------------------------------------------------------------------
function Fermer_Callback(hObject, eventdata, handles)
close('all')

% --------------------------------------------------------------------
function transf_fourier_Callback(hObject, eventdata, handles)
hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
           'Position',[.20 .1 .78 .87]);
imdata1 = imread('lena_color.tif'); 
imdata = rgb2gray(imdata1);  %%noir et blanc

%Get Fourier Transform of an image
F = fft2(imdata);
% Fourier transform of an image we can't see details
S = abs(F);

%get the centered spectrum just point
Fsh = fftshift(F);

%apply log transform
S2 = log(1+abs(Fsh));

% pour recupir?e l'iamage  l'inveerce de tf
F = ifftshift(Fsh);
f = ifft2(F);

subplot(2,3,1, 'Parent', hp);imshow(imdata1);title('Original Image');
subplot(2,3,2, 'Parent', hp);imshow(imdata);title('Gray Image');
subplot(2,3,3, 'Parent', hp);imshow(S,[]);title('Fourier transform of an image');
subplot(2,3,4, 'Parent', hp);imshow(abs(Fsh),[]);title('Centered fourier transform of Image');
subplot(2,3,5, 'Parent', hp);imshow(S2,[]);title('log transformed Image');
subplot(2,3,6, 'Parent', hp);imshow(f,[]);title('l inverce');


% --- Executes on key press with focus on rectangle and none of its controls.
function rectangle_KeyPressFcn(hObject, eventdata, handles)



% --- Executes when selected object is changed in uibuttongroup2.
function uibuttongroup2_SelectionChangedFcn(hObject, eventdata, handles)
% hObject    handle to the selected object in uibuttongroup2 
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
h=get(handles.uibuttongroup2,'SelectedObject');
get(h,'Tag')
switch get(h,'Tag')
    
    case 'rectangle'
        hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
        Image= imread('coins.png');
        imshow(Image);
        
        h=Image;
        s= strel('rectangle',[5,5]);  % square
        i=imerode(h,s);
        j=imdilate(h,s);
        subplot(1,3,1, 'Parent', hp); imshow(h);title('image originale');
        subplot(1,3,2, 'Parent', hp); imshow(i);title('image erroder');
        subplot(1,3,3, 'Parent', hp); imshow(j);title('image delater');
case 'line'
        hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
        Image= imread('coins.png');
        imshow(Image);
        
        h=Image;
        s= strel('line',10,45);  % square
        i=imerode(h,s);
        j=imdilate(h,s);
        subplot(1,3,1, 'Parent', hp); imshow(h);title('image originale');
        subplot(1,3,2, 'Parent', hp); imshow(i);title('image erroder');
        subplot(1,3,3, 'Parent', hp); imshow(j);title('image delater');
    case 'square'
        hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
        Image= imread('coins.png');
        imshow(Image);
        
        h=Image;
        s= strel('square',11);  % square
        i=imerode(h,s);
        j=imdilate(h,s);
        subplot(1,3,1, 'Parent', hp); imshow(h);title('image originale');
        subplot(1,3,2, 'Parent', hp); imshow(i);title('image erroder');
        subplot(1,3,3, 'Parent', hp); imshow(j);title('image delater');
    case 'disk'
        hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
        Image= imread('coins.png');
        imshow(Image);
        
        h=Image;
        s= strel('disk',5);  % square
        i=imerode(h,s);
        j=imdilate(h,s);
        subplot(1,3,1, 'Parent', hp); imshow(h);title('image originale');
        subplot(1,3,2, 'Parent', hp); imshow(i);title('image erroder');
        subplot(1,3,3, 'Parent', hp); imshow(j);title('image delater');
    case 'ball'
        hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
        Image= imread('coins.png');
        imshow(Image);
        
        h=Image;
        s= strel('ball',15,5);  % square
        i=imerode(h,s);
        j=imdilate(h,s);
        subplot(1,3,1, 'Parent', hp); imshow(h);title('image originale');
        subplot(1,3,2, 'Parent', hp); imshow(i);title('image erroder');
        subplot(1,3,3, 'Parent', hp); imshow(j);title('image delater');
   case 'diamond'
        hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
        Image= imread('coins.png');
        imshow(Image);
        
        h=Image;
        s= strel('diamond',5);  % square
        i=imerode(h,s);
        j=imdilate(h,s);
        subplot(1,3,1, 'Parent', hp); imshow(h);title('image originale');
        subplot(1,3,2, 'Parent', hp); imshow(i);title('image erroder');
        subplot(1,3,3, 'Parent', hp); imshow(j);title('image delater');
    otherwise
        hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
        Image= imread('cameraman.tif');
        subplot(1,3,1, 'Parent', hp); imshow(Image);title('otherwise');
        subplot(1,3,2, 'Parent', hp); imshow(Image);title('otherwise');
        subplot(1,3,3, 'Parent', hp); imshow(Image);title('otherwise');
end


% --------------------------------------------------------------------
function contours_Actif_Callback(hObject, eventdata, handles)
% hObject    handle to contours_Actif (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
I = imread('coins.png');
imshow(I)
hold on
title('Image Originale');
whos I
mask = false(size(I));
mask(1:246,1:300) = true;
visboundaries(mask,'Color','b');
bw = activecontour(I, mask, 8000, 'edge');
visboundaries(bw,'Color','r'); 
title(' contour Initial (blue) and contour final (red)');
figure, imshow(bw)
title(' Image segment?e ');



% --------------------------------------------------------------------
function Souvola_Callback(hObject, eventdata, handles)
% hObject    handle to Souvola (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
 hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
Image = imread('lena_color.tif');
output = sauvola(Image, 351, 0.4, 128);  
subplot(1,2,1, 'Parent', hp); imshow(Image);title('Image originale');
subplot(1,2,2, 'Parent', hp); imshow(output);title('Image avec souvola');


% --------------------------------------------------------------------
function Dicom_Callback(hObject, eventdata, handles)
% hObject    handle to Dicom (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
[filename,pathname]=uigetfile({'*.*'},'Select file');
    Filename=fullfile(pathname,filename);
var=strcat(pathname,filename);
[ORI_IMG,map]=dicomread(var);
axis(handles.axesTest);
imshow(ORI_IMG,map);
title('Dicom image');

% --------------------------------------------------------------------
function Kirsch_Callback(hObject, eventdata, handles)
% hObject    handle to Kirsch (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
hp = uipanel('Title','Main Panel','FontSize',12,...
            'BackgroundColor','white',...
            'Position',[.20 .1 .78 .87]);
X=imread('coins.png');
X = mat2gray(X);

f = im2double(X);
h1 = (1/15)*[5 5 5;-3 0 -3; -3 -3 -3];
Y1 = convn(f, h1);
Y1 = imresize(Y1,[246 300]);
Y=mat2gray(Y1);
 
subplot(1,2,1, 'Parent', hp); imshow(X);title('Image originale');
subplot(1,2,2, 'Parent', hp); imshow(Y);title('Image avec kirsh');
