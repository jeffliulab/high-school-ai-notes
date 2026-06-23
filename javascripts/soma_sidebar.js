/* soma-zero 风格侧栏手风琴：保持当前页所在章节常开，
   展开其他章节时收起其余非活动章节（活动章节始终保持展开）。
   Material 的左侧导航用 <input type="checkbox"> 控制展开/收起。 */
document$.subscribe(() => {
  const toggles = document.querySelectorAll(
    '.md-sidebar--primary .md-nav__toggle'
  );
  if (!toggles.length) return;

  const activeLink = document.querySelector(
    '.md-sidebar--primary .md-nav__link--active'
  );
  const activeItem = activeLink ? activeLink.closest('.md-nav__item--nested') : null;
  const activeToggle = activeItem ? activeItem.querySelector(':scope > .md-nav__toggle') : null;

  toggles.forEach((toggle) => {
    toggle.addEventListener('change', function () {
      if (!this.checked) return;
      toggles.forEach((other) => {
        if (other !== this && other !== activeToggle) other.checked = false;
      });
    });
  });
});
