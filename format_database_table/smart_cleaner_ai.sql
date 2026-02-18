-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: localhost
-- Generation Time: Feb 18, 2026 at 10:16 AM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.2.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `smart_cleaner_ai`
--

-- --------------------------------------------------------

--
-- Table structure for table `file_history`
--

CREATE TABLE `file_history` (
  `id` int(11) NOT NULL,
  `user_id` int(11) NOT NULL,
  `original_filename` varchar(255) DEFAULT NULL,
  `cleaned_filename` varchar(255) DEFAULT NULL,
  `file_path` varchar(500) DEFAULT NULL,
  `total_rows` int(11) DEFAULT 0,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `file_history`
--

INSERT INTO `file_history` (`id`, `user_id`, `original_filename`, `cleaned_filename`, `file_path`, `total_rows`, `created_at`) VALUES
(1, 1, 'bookstore_data_dirty.csv', 'clean_bookstore_data_dirty.csv', 'temp_files/bookstore_data_dirty.csv', 0, '2026-02-07 06:55:46'),
(2, 1, 'bookstore_data_dirty.csv', 'clean_bookstore_data_dirty.csv', 'temp_files/bookstore_data_dirty.csv', 0, '2026-02-07 07:11:32'),
(3, 1, 'used_car_price_dataset_extended.csv', 'clean_used_car_price_dataset_extended.csv', 'temp_files/used_car_price_dataset_extended.csv', 0, '2026-02-12 07:45:47'),
(4, 1, 'used_car_price_dataset_extended.csv', 'clean_used_car_price_dataset_extended.csv', 'temp_files/used_car_price_dataset_extended.csv', 0, '2026-02-12 07:48:24'),
(5, 1, 'used_car_price_dataset_extended.csv', 'clean_used_car_price_dataset_extended.csv', 'temp_files/used_car_price_dataset_extended.csv', 0, '2026-02-17 07:03:56'),
(6, 1, 'used_car_price_dataset_extended.csv', 'clean_used_car_price_dataset_extended.csv', 'temp_files/used_car_price_dataset_extended.csv', 0, '2026-02-17 07:05:09');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `username` varchar(50) NOT NULL,
  `email` varchar(100) NOT NULL,
  `password_hash` varchar(255) NOT NULL,
  `role` enum('user','admin') DEFAULT 'user',
  `plan` enum('free','pro') DEFAULT 'free',
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `last_login` timestamp NULL DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `username`, `email`, `password_hash`, `role`, `plan`, `created_at`, `last_login`) VALUES
(1, 'admin', '67160162@go.buu.ac.th', '$pbkdf2-sha256$29000$2Fsr5dxbKyXE2JvTOsfYew$djDD7gFNqBXbOxxhGEgHbwnA6dPDbV0zuVjIOh7NujU', 'admin', 'pro', '2026-02-07 02:24:23', '2026-02-17 06:57:57'),
(2, 'tester', 'tester@gmail.com', '$pbkdf2-sha256$29000$x7j33jvn/B9DaM055xzDuA$h2Zvkzpq0srWBFpZS9purK286CvMdymlS8.P4e4Hsjw', 'user', 'free', '2026-02-07 02:23:37', '2026-02-07 06:57:29');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `file_history`
--
ALTER TABLE `file_history`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `file_history`
--
ALTER TABLE `file_history`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=7;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `file_history`
--
ALTER TABLE `file_history`
  ADD CONSTRAINT `file_history_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
