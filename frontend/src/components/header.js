import React, { useState } from "react";
import { Link } from "react-router-dom";
export default function Header() {
  const [open, setOpen] = useState(false);

  const toggleMenu = () => setOpen((prev) => !prev);

  return (
    <>
      <header className="header">
        <div className="logo">
          Deep<span>Scan</span>
        </div>
        <div className={`menu-icon ${open ? "open" : ""}`} onClick={toggleMenu}>
          <span />
          <span />
          <span />
        </div>
      </header>
      <aside className={`side-nav ${open ? "open" : ""}`}>
        <nav>
          <ul>
            <li>
              <Link to="/" onClick={toggleMenu}>
                Home
              </Link>
            </li>
            <li>
              <Link to="/scanner" onClick={toggleMenu}>
                Scanner
              </Link>
            </li>
            <li>
              <Link to="/about" onClick={toggleMenu}>
                About Us
              </Link>
            </li>
            <li>
              <Link to="/contact" onClick={toggleMenu}>
                Contact
              </Link>
            </li>
          </ul>
        </nav>
      </aside>
    </>
  );
}
