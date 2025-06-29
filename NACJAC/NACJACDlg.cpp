
// NACJACDlg.cpp : implementation file
//
#include <tchar.h> 

#include "pch.h"
#include "framework.h"
#include "NACJAC.h"
#include "NACJACDlg.h"
#include "afxdialogex.h"
#include "random.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

struct RandData {
	int lower;
	int upper;
	int result;
	bool wildcard;
};

// CNACJACDlg dialog
CNACJACDlg::CNACJACDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_NACJAC_DIALOG, pParent)
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CNACJACDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CNACJACDlg, CDialogEx)
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED( IDC_ADD, &CNACJACDlg::OnBnClickedAddButton )
	ON_BN_CLICKED( IDC_GO, &CNACJACDlg::OnBnClickedGoButton )
	ON_BN_CLICKED( IDC_CLEAR, &CNACJACDlg::OnBnClickedClearButton )
	ON_LBN_SELCHANGE( IDC_LIST5, &CNACJACDlg::OnLbnSelchangeListBox )
	ON_WM_CTLCOLOR()
END_MESSAGE_MAP()


// CNACJACDlg message handlers

BOOL CNACJACDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// Set the icon for this dialog.  The framework does this automatically
	//  when the application's main window is not a dialog
	SetIcon(m_hIcon, TRUE);			// Set big icon
	SetIcon(m_hIcon, FALSE);		// Set small icon

	HICON hIcon = AfxGetApp()->LoadIcon( IDI_ICON1 );

	// Assign the icon to the button
	CButton* pButton = (CButton*)GetDlgItem( IDC_BUTTON1 );
	pButton->SetIcon( hIcon );

	CEdit* pState = (CEdit*)GetDlgItem( IDC_STATE );
	CEdit* pLower = (CEdit*)GetDlgItem( IDC_LOWER_BOUND );
	CEdit* pUpper = (CEdit*)GetDlgItem( IDC_UPPER_BOUND );
	CEdit* pResult = (CEdit*)GetDlgItem( IDC_RESULT );

	pState->ModifyStyle( 0, ES_NUMBER );
	pLower->ModifyStyle( 0, ES_NUMBER );
	pUpper->ModifyStyle( 0, ES_NUMBER );
	pResult->ModifyStyle( 0, ES_NUMBER );

	// TODO: Add extra initialization here
	SetWindowText( TITLE );

	GetDlgItem( IDC_EDIT1 )->SetFocus();

	SetWindowTheme( GetDlgItem( IDC_CHECK1 )->GetSafeHwnd(), L" ", L" ");

	#ifdef _DEBUG
		AllocConsole();
		freopen_s( (FILE**)stdout, "CONOUT$", "w", stdout );
	#endif // _DEBUG

	return FALSE;  // return TRUE  unless you set the focus to a control
}

// If you add a minimize button to your dialog, you will need the code below
//  to draw the icon.  For MFC applications using the document/view model,
//  this is automatically done for you by the framework.

void CNACJACDlg::OnPaint()
{
	CPaintDC dc( this ); // device context for painting

	if (IsIconic())
	{
		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// Draw the icon
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}

	// Load the bitmap resource
	CBitmap bitmap;
	bitmap.LoadBitmap( IDB_BITMAP2 ); // IDB_BACKGROUND is your resource ID

	// Select the bitmap into a memory DC
	CDC memDC;
	memDC.CreateCompatibleDC( &dc );
	CBitmap* pOldBitmap = memDC.SelectObject( &bitmap );

	// Get the client area dimensions
	CRect clientRect;
	GetClientRect( &clientRect );

	// Get the bitmap dimensions
	BITMAP bmp;
	bitmap.GetBitmap( &bmp );

	// Draw the bitmap stretched to cover the dialog
	dc.StretchBlt( 0, 0, clientRect.Width(), clientRect.Height(),
		&memDC, 0, 0, bmp.bmWidth, bmp.bmHeight, SRCCOPY );

	// Restore the old bitmap
	memDC.SelectObject( pOldBitmap );
}

// The system calls this function to obtain the cursor to display while the user drags
//  the minimized window.
HCURSOR CNACJACDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

void CNACJACDlg::OnLbnSelchangeListBox() 
{
	// Get the currently selected item
	CListBox* pListBox = (CListBox*)GetDlgItem( IDC_LIST5 );
	int idx = pListBox->GetCurSel();

	if (idx != LB_ERR) { // Ensure an item is selected
		RandData* r = (RandData*)pListBox->GetItemDataPtr( idx );
		if (r != nullptr) {
			delete r;
		}

		pListBox->DeleteString( idx ); // Remove the selected item
	}
}

void CNACJACDlg::OnBnClickedAddButton()
{
	CListBox* pListBox = (CListBox*)GetDlgItem( IDC_LIST5 );

	CButton* pWildcard = (CButton*)GetDlgItem( IDC_CHECK1 );
	if (pWildcard != nullptr) {
		int state = pWildcard->GetCheck();

		if (state == BST_CHECKED) {
			RandData* r = new RandData{ 0, 0, 0, true };

			int idx = pListBox->AddString( L"[Wildcard]" );
			if (idx != LB_ERR) {
				pListBox->SetItemDataPtr( idx, r );
			}

			return;
		}
	}

	CEdit* pLower = (CEdit*)GetDlgItem( IDC_LOWER_BOUND );
	CEdit* pUpper = (CEdit*)GetDlgItem( IDC_UPPER_BOUND );
	CEdit* pResult = (CEdit*)GetDlgItem( IDC_RESULT );

	CString lower_str;
	CString upper_str;
	CString result_str;

	pLower->GetWindowText( lower_str );
	pUpper->GetWindowText( upper_str );
	pResult->GetWindowText( result_str );


	if (lower_str.IsEmpty()) {
		AfxMessageBox( L"Lower bound value missing", MB_OK );
		return;
	}
	else if (upper_str.IsEmpty()) {
		AfxMessageBox( L"Upper bound value missing", MB_OK );
		return;
	}
	else if (result_str.IsEmpty()) {
		AfxMessageBox( L"Result value missing", MB_OK );
		return;
	}

	int lower = _tcstol( lower_str, nullptr, 10 );
	int upper = _tcstol( upper_str, nullptr, 10 );
	int result = _tcstol( result_str, nullptr, 10 );

	//printf( "%ull %i %i %i\n", state, lower, upper, result );

	if (lower > upper || (result > upper || result < lower)) {
		AfxMessageBox( L"Invalid bound values", MB_OK );
		return;
	}

	RandData* r = new RandData{ lower, upper, result, false };

	int idx = pListBox->AddString( L"math.random(" + lower_str + L", " + upper_str + L") == " + result_str );
	if (idx != LB_ERR) {
		pListBox->SetItemDataPtr( idx, r );
	}
}

void CNACJACDlg::OnBnClickedGoButton()
{
	CListBox* pListBox = (CListBox*)GetDlgItem( IDC_LIST5 );
	CEdit* pState = (CEdit*)GetDlgItem( IDC_STATE );
	CString state_str;
	pState->GetWindowText( state_str );

	if (state_str.IsEmpty()) {
		AfxMessageBox( L"State value missing", MB_OK );
		return;
	}

	uint64_t state = _tcstoui64( state_str, nullptr, 10 );

	int count = pListBox->GetCount();
	RandData** rList = new RandData*[count];

	for (int i = 0; i < count; ++i) {
		RandData* r = (RandData*)pListBox->GetItemDataPtr( i );
		if (r != nullptr) {
			rList[i] = r;
		}
	}

	uint64_t ostate = state;
	int calls = 0;

	while (true) {
		uint64_t currState = ostate;
		bool found = true;

		for (int i = 0; i < count; ++i) {
			RandData* r = rList[i];
			if (r->wildcard) {
				pcg32_random( &currState );
				continue;
			}

			int lower = r->lower;
			int upper = r->upper;
			int expectedResult = r->result;

			int result = math_random( &currState, lower, upper );
			if (result != expectedResult) {
				found = false;
				break;
			}
		}

		if (!found) {
			pcg32_random( &ostate );
			calls++;
		}
		else {
			break;
		}
	}
	delete[] rList;

	CString successMessage;
	successMessage.Format( L"%d calls required to reach desired outcome", calls );

	AfxMessageBox( successMessage, MB_ICONINFORMATION | MB_OK );
}

void CNACJACDlg::OnBnClickedClearButton()
{
	CListBox* pListBox = (CListBox*)GetDlgItem( IDC_LIST5 );

	int count = pListBox->GetCount();
	for (int i = 0; i < count; ++i) {
		RandData* r = (RandData*)pListBox->GetItemDataPtr( i );
		if (r != nullptr) {
			delete r;
		}
	}

	pListBox->ResetContent();
}

HBRUSH CNACJACDlg::OnCtlColor( CDC* pDC, CWnd* pWnd, UINT nCtlColor )
{
	HBRUSH hbr = CDialog::OnCtlColor( pDC, pWnd, nCtlColor );

	int id = pWnd->GetDlgCtrlID();
	if (id == IDC_LIST5)
	{
		hbr = CreateSolidBrush( RGB( 255, 255, 255 ) );
		//pDC->SetTextColor( RGB( 1, 237, 237 ) );     //Try changing the colour of the text
		pDC->SetBkMode( TRANSPARENT );
	}
	else if (id == IDC_ADD || id == IDC_CLEAR || id == IDC_GO) {
		hbr = CreateSolidBrush( RGB( 190, 190, 190 ) );
		//pDC->SetTextColor( RGB( 1, 237, 237 ) );     //Try changing the colour of the text
		pDC->SetBkMode( TRANSPARENT );
	}
	else if(id != IDC_STATE && id != IDC_LOWER_BOUND && id != IDC_UPPER_BOUND && id != IDC_RESULT){
		pDC->SetBkMode( TRANSPARENT );

		return (HBRUSH)::GetStockObject( NULL_BRUSH );
	}

	/*// TODO: Change any attributes of the DC here
	if (pWnd->GetDlgCtrlID() == IDC_GROUPBOX1)
	{
		pDC->SetTextColor( RGB( 255, 0, 0 ) );     //Try changing the colour of the text
		pDC->SetBkMode( TRANSPARENT );
		hbr = (HBRUSH)m_brush;     //or the background brush, here m_brushReadOnly is a CBrush, intiialised in the OnInitDialog
	}*/


	// TODO: Return a different brush if the default is not desired
	return hbr;
}