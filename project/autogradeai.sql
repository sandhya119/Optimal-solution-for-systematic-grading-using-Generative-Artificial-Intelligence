-- phpMyAdmin SQL Dump
-- version 5.0.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Dec 09, 2024 at 03:35 PM
-- Server version: 10.4.11-MariaDB
-- PHP Version: 7.4.6

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `autogradeai`
--

-- --------------------------------------------------------

--
-- Table structure for table `tbl_admin`
--

CREATE TABLE `tbl_admin` (
  `id` int(10) NOT NULL,
  `username` varchar(100) NOT NULL,
  `pass` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `tbl_admin`
--

INSERT INTO `tbl_admin` (`id`, `username`, `pass`) VALUES
(1, 'admin', '123');

-- --------------------------------------------------------

--
-- Table structure for table `tbl_external`
--

CREATE TABLE `tbl_external` (
  `id` int(100) NOT NULL,
  `usn` varchar(100) NOT NULL,
  `aiml` int(100) NOT NULL,
  `aiml_g` varchar(100) NOT NULL,
  `java` int(100) NOT NULL,
  `java_g` varchar(100) NOT NULL,
  `dsa` int(100) NOT NULL,
  `dsa_g` varchar(100) NOT NULL,
  `c` int(100) NOT NULL,
  `c_g` varchar(100) NOT NULL,
  `my_sql` int(100) NOT NULL,
  `sql_g` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- --------------------------------------------------------

--
-- Table structure for table `tbl_feedback`
--

CREATE TABLE `tbl_feedback` (
  `id` int(100) NOT NULL,
  `feedback` varchar(5000) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `tbl_feedback`
--

INSERT INTO `tbl_feedback` (`id`, `feedback`) VALUES
(1, 'Excellent in lab, improve theory and assignments.');

-- --------------------------------------------------------

--
-- Table structure for table `tbl_internal`
--

CREATE TABLE `tbl_internal` (
  `id` int(11) NOT NULL,
  `usn` int(11) NOT NULL,
  `subject` varchar(100) NOT NULL,
  `internal1` int(11) NOT NULL,
  `internal2` int(11) NOT NULL,
  `internal3` int(11) NOT NULL,
  `lab` int(11) NOT NULL,
  `assignment` int(11) NOT NULL,
  `s_discipline` int(11) NOT NULL,
  `s_report` int(11) NOT NULL,
  `s_presentation` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `tbl_internal`
--

INSERT INTO `tbl_internal` (`id`, `usn`, `subject`, `internal1`, `internal2`, `internal3`, `lab`, `assignment`, `s_discipline`, `s_report`, `s_presentation`) VALUES
(1, 4, 'AIML', 15, 18, 16, 8, 9, 3, 4, 10),
(2, 4, 'AIML', 13, 15, 16, 7, 4, 5, 7, 8),
(3, 4, 'AIML', 19, 16, 18, 9, 10, 3, 5, 12),
(4, 4, 'DSA', 18, 17, 16, 9, 10, 3, 5, 12),
(5, 4, 'SQL', 12, 15, 20, 20, 10, 20, 13, 13);

-- --------------------------------------------------------

--
-- Table structure for table `tbl_reg_std`
--

CREATE TABLE `tbl_reg_std` (
  `id` int(10) NOT NULL,
  `std_name` varchar(1000) NOT NULL,
  `usn` varchar(10) NOT NULL,
  `mob` varchar(10) NOT NULL,
  `mail` varchar(50) NOT NULL,
  `branch` varchar(50) NOT NULL,
  `year` varchar(10) NOT NULL,
  `pass` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `tbl_reg_std`
--

INSERT INTO `tbl_reg_std` (`id`, `std_name`, `usn`, `mob`, `mail`, `branch`, `year`, `pass`) VALUES
(1, 'Ananya', '4CB21CB038', '9418689135', 'ananya@gmail.com', 'CSBS', '2021', '123'),
(2, 'Varsha', '4CB21CB009', '9764356778', 'varsha@gmail.com', 'CSBS', '2021', '3456');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `tbl_admin`
--
ALTER TABLE `tbl_admin`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `tbl_external`
--
ALTER TABLE `tbl_external`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `tbl_feedback`
--
ALTER TABLE `tbl_feedback`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `tbl_internal`
--
ALTER TABLE `tbl_internal`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `tbl_reg_std`
--
ALTER TABLE `tbl_reg_std`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `tbl_admin`
--
ALTER TABLE `tbl_admin`
  MODIFY `id` int(10) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `tbl_external`
--
ALTER TABLE `tbl_external`
  MODIFY `id` int(100) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `tbl_feedback`
--
ALTER TABLE `tbl_feedback`
  MODIFY `id` int(100) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT for table `tbl_internal`
--
ALTER TABLE `tbl_internal`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;

--
-- AUTO_INCREMENT for table `tbl_reg_std`
--
ALTER TABLE `tbl_reg_std`
  MODIFY `id` int(10) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
