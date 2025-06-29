
// NACJACDlg.h : header file
//

#pragma once

// CNACJACDlg dialog
class CNACJACDlg : public CDialogEx
{
// Construction
public:
	CNACJACDlg(CWnd* pParent = nullptr);	// standard constructor

// Dialog Data
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_NACJAC_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support


// Implementation
protected:
	HICON m_hIcon;

	// Generated message map functions
	virtual BOOL OnInitDialog();
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	afx_msg HBRUSH OnCtlColor( CDC* pDC, CWnd* pWnd, UINT nCtlColor );
	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnLbnSelchangeListBox();
	afx_msg void OnBnClickedAddButton();
	afx_msg void OnBnClickedGoButton();
	afx_msg void OnBnClickedClearButton();

private:
	CBrush m_brush; // Brush for custom background color
};
