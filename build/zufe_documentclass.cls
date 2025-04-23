\ProvidesClass{zufe_documentclass}

% 加载基础class，设置相关参数
\LoadClass[UTF8,a4paper,12pt,oneside,fontset=mac,openany]{ctexbook}

% 设置页面边距
\RequirePackage{geometry}
\geometry{
  a4paper,
  top=3.5cm, 
  bottom=2.5cm, 
  left=2.5cm, 
  right=2.5cm,
  headheight=2.5cm,
  footskip=2cm,
  bindingoffset=0cm
}
\setlength{\oddsidemargin}{0pt} % 奇数页左边距偏移
\setlength{\evensidemargin}{0pt} % 偶数页左边距偏移

% 设置字体
\setCJKmainfont{SimSong} % 设置默认中文字体为宋体
\setmainfont{Times New Roman} % 设置默认英文字体为Times New Roman

% 判断是否有副标题，默认没有
\newif\ifhaveSubTitle\haveSubTitlefalse
% 有副标题则将值设为true，显示副标题
\newcommand{\have@SubTitle}{}
\newcommand{\haveSub}[1]{\haveSubTitletrue\renewcommand{\have@SubTitle}{#1}}

% 字号定义
% 定义中文排版常用字号命令，每个命令包含两个参数：
% 第一个参数是字号大小，第二个参数是行距（通常为字号的1.2倍）
\DeclareRobustCommand{\chuhao}{\fontsize{42pt}{50.4pt}\selectfont} % 初号
\DeclareRobustCommand{\xiaochu}{\fontsize{36pt}{43.2pt}\selectfont} % 小初
\DeclareRobustCommand{\yihao}{\fontsize{26pt}{31.2pt}\selectfont} % 一号
\DeclareRobustCommand{\xiaoyi}{\fontsize{24pt}{28.8pt}\selectfont} % 小一
\DeclareRobustCommand{\erhao}{\fontsize{22pt}{26.4pt}\selectfont} % 二号
\DeclareRobustCommand{\xiaoer}{\fontsize{18pt}{21.6pt}\selectfont} % 小二
\DeclareRobustCommand{\sanhao}{\fontsize{16pt}{19.2pt}\selectfont} % 三号
\DeclareRobustCommand{\xiaosan}{\fontsize{15pt}{18pt}\selectfont} % 小三
\DeclareRobustCommand{\sihao}{\fontsize{14pt}{16.8pt}\selectfont} % 四号
\DeclareRobustCommand{\shisanbang}{\fontsize{13pt}{15.pt}\selectfont} % 13磅
\DeclareRobustCommand{\xiaosi}{\fontsize{12pt}{14.4pt}\selectfont} % 小四
\DeclareRobustCommand{\wuhao}{\fontsize{10.5pt}{12.6pt}\selectfont} % 五号
\DeclareRobustCommand{\xiaowu}{\fontsize{9pt}{10.8pt}\selectfont} % 小五
\DeclareRobustCommand{\liuhao}{\fontsize{7.5pt}{9pt}\selectfont} % 六号
\DeclareRobustCommand{\xiaoliu}{\fontsize{6.5pt}{7.8pt}\selectfont} % 小六
\DeclareRobustCommand{\qihao}{\fontsize{5.5pt}{6.6pt}\selectfont} % 七号
\DeclareRobustCommand{\bahao}{\fontsize{5pt}{6pt}\selectfont} % 八号

