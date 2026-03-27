" --- General Settings ---
set number              " Show line numbers
set mouse=a             " Enable mouse support in all modes
set clipboard=unnamedplus " Use system clipboard (requires vim-gtk or similar)
set nocursorline          " Highlight the current line
set showmatch           " Highlight matching brackets

" --- Search ---
set hlsearch            " Highlight search results
set incsearch           " Show search matches as you type
set ignorecase          " Ignore case when searching...
set smartcase           " ...unless the search query contains an uppercase letter

" --- Python & Bash Specifics ---
filetype plugin indent on  " Detect file types and load specific indentation
let python_highlight_all = 1 " Enhanced Python highlighting

" --- Tabs and Indentation ---
set expandtab           " Convert tabs to spaces
set shiftwidth=4        " Number of spaces for auto-indent
set softtabstop=4       " Number of spaces per tab while editing
set tabstop=4           " Number of spaces a tab counts for
set autoindent          " Copy indent from current line when starting a new one
set smartindent         " Be smart about indentation (e.g., after '{')

" --- UI and Performance ---
syntax on               " Enable syntax highlighting
set hidden              " Allow switching buffers without saving
set noswapfile          " Disable swap files (prevents 'ATTENTION' messages)
set updatecount=0       " Disable swap file writing
set updatetime=300      " Faster completion and diagnostic updates

" --- Better Syntax Colors ---
" If your terminal supports it, this is a clean, built-in color scheme
" You can change 'desert' to 'ron', 'slate', or 'industry'
colorscheme desert

" --- Custom Keybindings ---
let mapleader = " "     " Set Space as your leader key

" Clear search highlighting with <Leader> + /
nnoremap <leader>/ :nohlsearch<CR>

" Fast saving with <Leader> + w
nnoremap <leader>w :w<CR>

" Use kj to exit Insert mode (much faster than Esc)
inoremap kj <Esc>


" --- Change cursor shape for Insert Mode ---
" 1 or 2 = Block
" 3 or 4 = Underline
" 5 or 6 = Vertical Bar

let &t_SI = "\e[5 q" " SI = Start Insert (Vertical Bar)
let &t_SR = "\e[3 q" " SR = Start Replace (Underline)
let &t_EI = "\e[1 q" " EI = End Insert (Back to Block)

set autochdir
