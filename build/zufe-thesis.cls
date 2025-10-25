%% zufe-thesis.cls
%% 浙江财经大学学位论文文档类
%% 版本: 1.0
%% 日期: 2025-01

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{zufe-thesis}[2025/01/16 v1.0 ZJUFE Thesis Template]

% ==================== 选项定义 ====================
% 学位类型
\newif\ifmaster\masterfalse
\newif\ifdoctor\doctorfalse
\DeclareOption{master}{\mastertrue\doctorfalse}
\DeclareOption{doctor}{\doctortrue\masterfalse}

% 论文版本
\newif\ifblind\blindfalse
\newif\ifdefense\defensefalse
\newif\iffinal\finalfalse
\DeclareOption{blind}{\blindtrue\defensefalse\finalfalse}
\DeclareOption{defense}{\defensefalse\blindfalse\finaltrue} % defense实际上就是final但用于标识
\DeclareOption{final}{\finaltrue\blindfalse\defensefalse}

% 默认选项
\ExecuteOptions{master,final}

% 其他选项传递给ctexbook
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{ctexbook}}

\ProcessOptions\relax

% ==================== 加载基础文档类 ====================
\LoadClass[
  a4paper,
  zihao=-4,
  twoside,
  openany
]{ctexbook}

% ==================== 加载必需宏包 ====================
% 页面设置
\RequirePackage{geometry}
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{titletoc}
\RequirePackage[export]{adjustbox}

% 字体设置
\RequirePackage{fontspec}
\RequirePackage{xeCJK}
\RequirePackage{anyfontsize}

% 图表设置
\RequirePackage{graphicx}
\RequirePackage{caption}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{array}
\RequirePackage{xcolor}
\RequirePackage{multirow}
\RequirePackage{ragged2e}

% 引用和链接
\RequirePackage{hyperref}
\RequirePackage[
  backend=biber,
  style=gb7714-2015,
  citestyle=authoryear,
  sorting=nyvt,
  maxnames=1,
  minnames=1,
  maxbibnames=99,
  doi=false,
  gbpub=false
]{biblatex}

% 其他工具
\RequirePackage{etoolbox}
\RequirePackage{indentfirst}
\RequirePackage{setspace}
\RequirePackage{tikz}

% ==================== 页面设置 ====================
\geometry{
  a4paper,
  top=3cm,
  bottom=2cm,
  left=3cm,
  right=3cm,
  headheight=2.5cm,
  footskip=1cm,
  bindingoffset=0cm
}

% 清除奇偶页边距偏移
% \setlength{\oddsidemargin}{0pt}
% \setlength{\evensidemargin}{0pt}

% ==================== 字体设置区域 ====================
% 用户可以在主文档中使用以下命令覆盖字体设置:
% \setCJKmainfont[Path=./]{你的宋体文件.ttf}
% \setCJKsansfont[Path=./]{你的黑体文件.ttf}
% \setCJKmonofont[Path=./]{你的楷体文件.ttf}
% \setmainfont[Path=./]{你的英文字体.ttf}

% 默认字体设置（使用系统字体）
\setCJKmainfont{宋体-简}[
  AutoFakeBold=2.5,
  ItalicFont=Kai
]
\setCJKsansfont{黑体-简}[AutoFakeBold=2.5]
\setCJKmonofont{Kai}[AutoFakeBold=2.5]
\setmainfont{Times New Roman}

% 定义字体命令供用户使用
%\newcommand{\songti}{\CJKfamily{zhsong}}
%\newcommand{\heiti}{\CJKfamily{zhhei}}
%\newcommand{\kaishu}{\CJKfamily{zhkai}}

% ==================== 行距和段落 ====================
% 首行缩进
\setlength{\parindent}{2em}

% 段间距
\setlength{\parskip}{0pt}

% 行距设置为20磅（约1.5倍）
\linespread{1.5}
\setlength{\baselineskip}{20pt}

% ==================== 页眉页脚 ====================
\pagestyle{fancy}
\fancyhf{} % 清空

% 页眉内容根据学位类型变化
\ifmaster
  \fancyhead[C]{\zihao{5}\songti 浙江财经大学硕士学位论文}
\else
  \ifdoctor
    \fancyhead[C]{\zihao{5}\songti 浙江财经大学博士学位论文}
  \fi
\fi

% 页码居中
\fancyfoot[C]{\thepage}

% 页眉线宽度
\renewcommand{\headrulewidth}{0.4pt}

% 重定义plain样式（章首页）
\fancypagestyle{plain}{
  \fancyhf{}
  \ifmaster
    \fancyhead[C]{\zihao{5}\songti 浙江财经大学硕士学位论文}
  \else
    \ifdoctor
      \fancyhead[C]{\zihao{5}\songti 浙江财经大学博士学位论文}
    \fi
  \fi
  \fancyfoot[C]{\thepage}
  \renewcommand{\headrulewidth}{0.4pt}
}

% ==================== 章节标题格式 ====================
% ctexbook日期和摘要名称设置
\ctexset{
  today=big,
  abstractname={摘\quad 要}
}

% 章标题格式：黑体三号居中
\titleformat{\chapter}[block]
  {\centering\zihao{3}\heiti}
  {\thechapter}
  {1em}
  {}

% 章标题间距：段前2行，段后1行
\titlespacing{\chapter}
  {0pt}
  {2\baselineskip}
  {1\baselineskip}

% 节标题格式：黑体四号居中
\titleformat{\section}[block]
  {\centering\zihao{4}\heiti}
  {\thesection}
  {1em}
  {}

% 节标题间距：段前1行，段后1行
\titlespacing{\section}
  {0pt}
  {1\baselineskip}
  {1\baselineskip}

% 小节标题格式：黑体小四号左对齐空两格
\titleformat{\subsection}[hang]
  {\zihao{-4}\heiti}
  {\thesubsection}
  {1em}
  {}

% 小节标题间距：段前0.5行，段后0.5行
\titlespacing{\subsection}
  {2em}
  {0.5\baselineskip}
  {0.5\baselineskip}

% 无编号章标题格式（参考文献、致谢、声明等）
\titleformat{name=\chapter,numberless}[block]
  {\centering\zihao{3}\heiti}
  {}
  {0pt}
  {}

\titlespacing{name=\chapter,numberless}
  {0pt}
  {2\baselineskip}
  {1\baselineskip}

% ==================== 目录格式 ====================
% 目录标题
\renewcommand{\contentsname}{目\quad 录}

% 目录格式设置
\ctexset{
  chapter/name = {},
  chapter/number = \arabic{chapter},
}

\titlecontents{section}[2em]
  {\zihao{-4}\songti}
  {\thecontentslabel\quad}
  {}
  {\titlerule*[0.5pc]{.}\contentspage}

\titlecontents{subsection}[4em]
  {\zihao{-4}\songti}
  {\thecontentslabel\quad}
  {}
  {\titlerule*[0.5pc]{.}\contentspage}

% ==================== 图表格式 ====================
% 图编号格式：章-图
\renewcommand{\thefigure}{\thechapter-\arabic{figure}}
\renewcommand{\thetable}{\thechapter-\arabic{table}}

% 图标题格式：宋体五号，编号与标题间空格分隔
\captionsetup[figure]{
  font={small,rm},           % 五号字体，Times New Roman/宋体
  labelsep=space,            % 图号与图名用空格隔开
  skip=20pt,                 % 段后1行（20pt）
  position=bottom,           % 图的正下方
  justification=centering,   % 居中排列
  aboveskip=0pt,            % 段前0行
  belowskip=20pt            % 段后1行
}

% 表标题格式：宋体五号，编号与标题间空格分隔
\captionsetup[table]{
  font={small,rm},           % 五号字体
  labelsep=space,            % 表号与表名用空格隔开
  position=top,              % 表的正上方
  justification=centering,   % 居中排列
  aboveskip=20pt,           % 段前1行
  belowskip=0pt             % 段后0行
}

% 表格行间距：单倍行距
\renewcommand{\arraystretch}{1.0}

% 表格换行左对齐
\newcolumntype{L}{>{\RaggedRight\arraybackslash}X}

% 重定义booktabs三线表线宽
\setlength{\heavyrulewidth}{1.5pt}  % 上下两根线1.5磅
\setlength{\lightrulewidth}{0.75pt} % 表头下0.75磅

% ==================== 数学公式格式 ====================
% 公式编号格式：章-公式
\renewcommand{\theequation}{\thechapter.\arabic{equation}}

% 公式环境前后间距
\AtBeginDocument{
  \setlength{\abovedisplayskip}{6pt}
  \setlength{\belowdisplayskip}{6pt}
  \setlength{\abovedisplayshortskip}{3pt}
  \setlength{\belowdisplayshortskip}{3pt}
}

% ==================== 脚注格式 ====================
% 脚注每页重新编号
\RequirePackage[perpage]{footmisc}
\RequirePackage{tikz}  % 用于绘制圆圈

% 脚注字体：宋体五号
\renewcommand{\footnotesize}{\zihao{5}}

% 定义圆圈数字命令
\newcommand{\circled}[1]{%
  \tikz[baseline=(char.base)]{
    \node[shape=circle,draw,inner sep=0.5pt,minimum size=1.2em] (char) {\small #1};
  }%
}

% 脚注编号格式：上标带圆圈的数字
\renewcommand{\thefootnote}{\protect\circled{\arabic{footnote}}}

% 确保脚注标记为上标
\renewcommand{\@makefnmark}{\hbox{\textsuperscript{\normalfont\@thefnmark}}}

% 脚注正文中的标记不使用上标
\renewcommand{\@makefntext}[1]{%
  \parindent 2em%
  \noindent
  \makebox[0pt][r]{\@thefnmark\hspace{0.5em}}#1%
}

% ==================== 参考文献设置 ====================
% biblatex设置（保留用户原有配置）
\ExecuteBibliographyOptions{sortlocale=zh__pinyin}

% 删除url和librarycatalog字段
\DeclareSourcemap{
  \maps[datatype=bibtex]{
    \map{
      \step[fieldset=url, null]
    }
    \map{
      \step[fieldset=librarycatalog, null]
    }
  }
}

% ==================== 超链接设置 ====================
\hypersetup{
  pdflang=zh-CN,
  colorlinks=true,
  linkcolor=black,
  citecolor=black,
  urlcolor=black,
  bookmarksnumbered=true,
  bookmarksopen=true
}

% ==================== 用户信息定义 ====================
% 论文基本信息
\newcommand{\thesisTitle}[1]{\def\@thesisTitle{#1}}
\newcommand{\thesisTitleEN}[1]{\def\@thesisTitleEN{#1}}
\newcommand{\authorName}[1]{\def\@authorName{#1}}
\newcommand{\studentID}[1]{\def\@studentID{#1}}
\newcommand{\mentorName}[1]{\def\@mentorName{#1}}
\newcommand{\majorName}[1]{\def\@majorName{#1}}
\newcommand{\deptName}[1]{\def\@deptName{#1}}
\newcommand{\submitDate}[1]{\def\@submitDate{#1}}
\newcommand{\submitDateEN}[1]{\def\@submitDateEN{#1}}
\newcommand{\reviewDate}[1]{\def\@reviewDate{#1}}
\newcommand{\secrecyLevel}[1]{\def\@secrecyLevel{#1}}

% 默认值
\thesisTitle{论文题目}
\thesisTitleEN{Thesis Title in English}
\authorName{学生姓名}
\studentID{学号}
\mentorName{导师姓名}
\majorName{专业名称}
\deptName{学院名称}
\submitDate{\today}
\submitDateEN{\today}
\reviewDate{\today}
\secrecyLevel{}

% ==================== 封面生成 ====================
% 定义一个用于自动换行、居中且带下划线的宏
\newcommand{\AutoWrapUnderline}[2]{%
  \parbox[t]{#1}{% [t] 表示顶部对齐
    \centering
    \kaishu #2% 内容
    \vspace*{0.5ex}
    \hrule height 0.4pt  % 下划线
  }%
}

% 定义一个用于单行、居中且带下划线的宏
\newcommand{\SimpleUnderline}[2]{%
  \parbox[t]{#1}{%
    \centering
    \kaishu #2%
    \vspace*{0.5ex} % 增加安全间距
    \hrule height 0.4pt
  }%
}

\newcommand{\makethesiscover}{
  \begin{titlepage}
    \centering
    
    % === 右上角：学校代码 & 密级 ===
    \hfill
    \begin{tabular}[t]{@{}l@{}}
      \zihao{-4}\kaishu 学校代码：11482 \\
      \zihao{-4}\kaishu 密\quad\quad 级：\@secrecyLevel
    \end{tabular}
    \par
    
    \vspace{1.5cm}
    
    % === 校徽与校名图片 ===
    \begin{center}
      \includegraphics[height=2.21cm, keepaspectratio, valign=c]{images/logo/schoolLogo.png}
      \hspace{0.75cm}% 调整两张图之间的间距，如 1em、0.5cm 等
      \includegraphics[height=1.73cm, keepaspectratio, valign=c]{images/logo/schoolName.png}
    \end{center}
    
    \vspace{0.8cm}
    
    % === 英文校名 ===
    {\zihao{4}\rmfamily\linespread{1}\selectfont
     \textbf{ZHEJIANG UNIVERSITY OF FINANCE AND ECONOMICS}\par}
    
    % \vspace{1cm}
    
    % === 中文“硕士学位论文”（隶书，30pt，加粗）===
    {\fontsize{30}{36}\selectfont\bfseries
     \ifxetex
       \CJKfontspec{报隶-简}%
     \else
       \CJKfontspec{STLiti}%
     \fi
     \textbf{硕士学位论文}\par}
    
    \vspace{0.8cm}
    
    % === 英文“MASTER DISSERTATION” ===
    {\zihao{4}\rmfamily\linespread{1}\selectfont
     \textbf{MASTER DISSERTATION}\par}
    
    \vspace{2.8cm}
    
    % === 基本信息表格（含论文题目）===
    \begin{center}
      \zihao{-3}
      \renewcommand{\arraystretch}{1.5}
      \begin{tabular}{l c}
        \ifblind
          % 盲审版：隐藏个人信息，但保留题目
          \heiti 论文题目： & \AutoWrapUnderline{10cm}{\@thesisTitle} \\
          \heiti 学\quad \quad 院：& \SimpleUnderline{10cm}{\@deptName} \\
          \heiti 专\quad \quad 业： & \SimpleUnderline{10cm}{\@majorName} \\
          \heiti 送审时间： & \SimpleUnderline{10cm}{\@reviewDate} \\
        \else
          % 非盲审版
          \heiti 论文题目： & \AutoWrapUnderline{10cm}{\@thesisTitle} \\
          \heiti 作者姓名： & \SimpleUnderline{10cm}{\@authorName} \\
          \heiti 专\quad \quad 业： & \SimpleUnderline{10cm}{\@majorName} \\
          \heiti 所在学院： & \SimpleUnderline{10cm}{\@deptName} \\
          \heiti 指导教师： & \SimpleUnderline{10cm}{\@mentorName} \\
          \heiti 完成日期： & \SimpleUnderline{10cm}{\@submitDate} \\
        \fi
      \end{tabular}
    \end{center}
    
  \end{titlepage}
  \clearpage
}

% ==================== 声明页生成 ====================
% --- 声明页  ---
\newcommand{\makestatement}{
  \ifblind
    % 盲审版：什么都不做
  \else
    \clearpage\thispagestyle{empty}
    \vspace*{-6mm}
    \begin{center}
      \heiti\zihao{2}\textmd{声明及论文使用的授权}
    \end{center}
    \vspace{10mm}
    {\zihao{-3}\setlength{\parskip}{0.4em}\renewcommand{\baselinestretch}{1.41}\selectfont
    本人郑重声明所呈交的论文是我个人在导师的指导下独立完成的。除了文中特别加以标注和致谢的地方外，论文中不包含其他人已经发表或撰写的研究成果。
    \par\vspace{15mm}
    \begin{flushright}
      论文作者签名：\hspace{75mm}年\hspace{8mm}月\hspace{8mm}日
    \end{flushright}
    \par\vspace{40mm}
    本人同意浙江财经大学有关保留使用学位论文的规定，即：学校有权保留送交论文的复印件，允许论文被查阅和借阅；学校可以上网公布全部内容，可以采用影印、缩印或其他复制手段保存论文。
    \par\vspace*{15mm}
    \begin{flushright}
      论文作者签名：\hspace{75mm}年\hspace{8mm}月\hspace{8mm}日
    \end{flushright}}
    \clearpage
  \fi
}

% ==================== 中文扉页 ====================
\newcommand{\makechinesetitlepage}{
  \clearpage
  \thispagestyle{empty}
  
  \begin{center}
    
    % 顶部：硕士学位论文（黑体，小一）
    {\zihao{1}\heiti 硕士学位论文}

    \vspace*{\fill}

    % 中部：论文中文标题（黑体，小一，自动换行）
    \parbox{0.8\textwidth}{\centering
      \zihao{1}\heiti \@thesisTitle
    }
    
    \vspace*{\fill}
    
    % 底部：完成日期（楷体，四号）
    {\zihao{4}\kaishu \@submitDate}
  \end{center}
  
  \clearpage
}

% ==================== 英文扉页 ====================
\newcommand{\makeenglishtitlepage}{
  \clearpage
  \thispagestyle{empty}
  
  \begin{center}
    
    % 顶部：Master Dissertation（Times New Roman，小一）
    {\zihao{1}\rmfamily \textbf{Master Dissertation}}
    
    \vspace*{\fill}
    
    % 中部：英文标题（Times New Roman，小一，自动换行）
    \parbox{0.8\textwidth}{\centering
      \zihao{1}\rmfamily \textbf{\@thesisTitleEN}
    }
    
    \vspace*{\fill}
    
    % 底部：Completion Date（Times New Roman，四号）
    {\zihao{4}\rmfamily \textbf{\@submitDateEN}}
  \end{center}
  
  \clearpage
}

% ==================== 摘要环境 ====================
% 中文摘要
\newenvironment{abstract}{
  \clearpage
  %\phantomsection
  %\addcontentsline{toc}{chapter}{摘要}
  
  \begin{center}
    {\zihao{3}\heiti 摘\quad 要}
  \end{center}
  
  \vspace{1cm}
  
  \par\zihao{-4}\songti
}{
  \clearpage
}

% 中文关键词
\newcommand{\keywords}[1]{%
  \par\vspace{\baselineskip}% ← 关键修改：空一行
  \noindent{\zihao{-4}\heiti 关键词：}{\zihao{-4}\songti #1}%
}

% 英文摘要
\newenvironment{abstracten}{
  \clearpage
  %\phantomsection
  %\addcontentsline{toc}{chapter}{Abstract}
  % ABSTRACT 标题：三号、加粗、Times New Roman、2倍行距、段前2行、段后1行
  \begin{center}
    \zihao{3}\bfseries
    \renewcommand{\baselinestretch}{2}\selectfont
    ABSTRACT
    \vspace{1\baselineskip} % 段后1行（因2倍行距，1行≈2*20pt/2=20pt）
  \end{center}
  % 摘要正文：小四、Times New Roman、20磅行距
  \par
  \zihao{-4}
  \renewcommand{\baselinestretch}{1}\setlength{\baselineskip}{20pt}\selectfont
  \noindent\ignorespaces
}{
  \clearpage
}

% 英文关键词（自动小写、分号分隔、末尾无标点）
\newcommand{\keywordsen}[1]{%
  \par\vspace{\baselineskip}% ← 空一行，与中文一致
  \noindent{\zihao{-4}\bfseries Keywords: }{\zihao{-4} #1}%
}

% ==================== 致谢环境 ====================
\newenvironment{acknowledgement}{
  \ifblind
    % 盲审版省略致谢
  \else
    \clearpage
    \phantomsection
    \addcontentsline{toc}{chapter}{致谢}
    
    \begin{center}
      {\zihao{3}\heiti 致\quad 谢}
    \end{center}
    
    \vspace{1cm}
    
    \par\zihao{-4}\songti
  \fi
}{
  \ifblind
  \else
    \clearpage
  \fi
}

% ==================== 附录相关 ====================
% 重新定义附录
\renewcommand{\appendix}{
  \clearpage
  \setcounter{chapter}{0}
  \renewcommand{\thechapter}{\Alph{chapter}}     % A, B, C...
  
  % 重定义 ctex 的章节格式
  \ctexset{
    chapter={
      name={附录,},
      number=\Alph{chapter},
    }
  }
  
  % 修改编号格式以符合要求
  \renewcommand{\thefigure}{\thechapter\arabic{figure}}      % 图A1
  \renewcommand{\thetable}{\thechapter\arabic{table}}        % 表A1
  \renewcommand{\theequation}{\thechapter\arabic{equation}}  % A1（无点）
  
  % 修改章标题格式（如果要严格按小四号宋体）
  \titleformat{\chapter}[block]
    {\centering\zihao{-4}\songti}    % 改为小四号宋体
    {附录\thechapter}
    {1em}
    {}
  
  % 修改目录格式
  \titlecontents{chapter}[0em]
    {\zihao{-4}\songti}
    {\thecontentslabel\quad}
    {}
    {\titlerule*[0.5pc]{.}\contentspage}
}

% 发表论文列表环境（盲审版省略）
\newenvironment{publications}{
  \ifblind
    % 盲审版省略
  \else
    \clearpage
    \phantomsection
    \addcontentsline{toc}{chapter}{附录：作者在读期间发表的学术论文}
    
    \begin{center}
      {\zihao{3}\heiti 附录：作者在读期间发表的学术论文}
    \end{center}
    
    \vspace{1cm}
    
    \par\zihao{-4}\songti
  \fi
}{
  \ifblind
  \else
    \clearpage
  \fi
}

% ==================== 封底 ====================
\newcommand{\makebackcover}{
  \clearpage
  \thispagestyle{empty}
  
  \vspace*{\fill}
    
  \parbox{\textwidth}{%
    \hspace*{2em}% 首行空两格
    \zihao{-3}%
    \textit{浙江财经大学}
  }

  \vspace{0.25cm}

  \hrulefill
  
  \vspace{0.25cm}

  {\zihao{4}地址：浙江省杭州市下沙高教园区学源街18号}
    
  \vspace{0.5cm}
    
  {\zihao{4}邮编：310018}
    
  \vspace{0.5cm}

  {\zihao{4}电话：0571-86735317}

  \vspace{0.5cm}
    
  {\zihao{4}网址：http://www.zufe.edu.cn}
}

% ==================== 页码设置 ====================
% 前置部分使用罗马数字页码
\renewcommand{\frontmatter}{
  \cleardoublepage
  \pagenumbering{Roman}
  \renewcommand{\thepage}{\Roman{page}}
} % 修正：强制大写罗马页码

% 正文部分使用阿拉伯数字页码
\renewcommand{\mainmatter}{
  \cleardoublepage
  \pagenumbering{arabic}
}

% 后置部分继续使用阿拉伯数字页码
\renewcommand{\backmatter}{
  % 继续使用阿拉伯数字，不重置页码
}

\endinput